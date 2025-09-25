from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import os
from dotenv import load_dotenv
import logging
from transformers import T5Tokenizer, T5ForConditionalGeneration, pipeline
from sqlalchemy import create_engine, text, MetaData, inspect
import pandas as pd
import sqlparse
import re
from sqlparse import sql, tokens
import torch

load_dotenv()

app = FastAPI(title="OptimaX SQL Chat API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200", "http://localhost:4201"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logger = logging.getLogger(__name__)

class ChatMessage(BaseModel):
    message: str
    system_prompt: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    sql_query: Optional[str] = None
    error: Optional[str] = None

class TextToSQLEngine:
    def __init__(self):
        self.engine = None
        self.text_to_sql_model = None
        self.tokenizer = None
        self.table_schema = None
        self.initialize()

    def initialize(self):
        try:
            database_url = os.getenv("DATABASE_URL")
            if not database_url:
                raise ValueError("DATABASE_URL not set in environment")

            self.engine = create_engine(database_url)

            # Use a working T5-based SQL model
            try:
                # Try the working t5-base text-to-sql model
                model_name = "juierror/text-to-sql-with-table-schema"
                logger.info(f"Loading T5 Text-to-SQL model: {model_name}")
                self.tokenizer = T5Tokenizer.from_pretrained(model_name)
                self.text_to_sql_model = T5ForConditionalGeneration.from_pretrained(model_name)
                logger.info("T5 Text-to-SQL model loaded successfully")
            except Exception as model_error:
                logger.warning(f"Failed to load specialized model: {model_error}")
                try:
                    # Fallback to basic t5-small
                    logger.info("Trying t5-small as fallback")
                    self.tokenizer = T5Tokenizer.from_pretrained("t5-small")
                    self.text_to_sql_model = T5ForConditionalGeneration.from_pretrained("t5-small")
                    logger.info("t5-small loaded successfully")
                except Exception as fallback_error:
                    logger.error(f"Failed to load any T5 model: {fallback_error}")
                    # Use rule-based approach only
                    self.tokenizer = None
                    self.text_to_sql_model = None
                    logger.info("Using rule-based approach only")

            # Get table schema information
            self.load_table_schema()

            logger.info("Text-to-SQL engine initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize Text-to-SQL engine: {str(e)}")
            raise

    def load_table_schema(self):
        """Load detailed table schema with column types"""
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text("""
                    SELECT column_name, data_type, is_nullable
                    FROM information_schema.columns
                    WHERE table_name = 'us_accidents'
                    ORDER BY ordinal_position
                """))

                schema_info = []
                for row in result:
                    col_name, data_type, nullable = row
                    schema_info.append(f"{col_name} ({data_type})")

                self.table_schema = {
                    "table_name": "us_accidents",
                    "columns": schema_info,
                    "schema_text": f"CREATE TABLE us_accidents (\n  {', '.join(schema_info)}\n);"
                }

                logger.info(f"Loaded schema for table: us_accidents with {len(schema_info)} columns")

        except Exception as e:
            logger.error(f"Failed to load table schema: {str(e)}")
            raise

    def validate_sql(self, sql_query: str) -> tuple[bool, str]:
        """Validate SQL for safety and syntax"""
        try:
            # Parse SQL
            parsed = sqlparse.parse(sql_query)
            if not parsed:
                return False, "Invalid SQL syntax"

            # Check for dangerous operations
            dangerous_keywords = ['DROP', 'DELETE', 'UPDATE', 'ALTER', 'INSERT', 'CREATE', 'TRUNCATE']
            sql_upper = sql_query.upper()

            for keyword in dangerous_keywords:
                if re.search(rf'\b{keyword}\b', sql_upper):
                    return False, f"Dangerous operation detected: {keyword}"

            # Ensure it's a SELECT statement
            if not sql_upper.strip().startswith('SELECT'):
                return False, "Only SELECT statements are allowed"

            # Add LIMIT if not present and query might return large results
            if 'LIMIT' not in sql_upper and ('GROUP BY' in sql_upper or 'ORDER BY' in sql_upper):
                if not sql_query.rstrip().endswith(';'):
                    sql_query = sql_query.rstrip() + ' LIMIT 1000;'
                else:
                    sql_query = sql_query.rstrip(';') + ' LIMIT 1000;'

            return True, sql_query

        except Exception as e:
            return False, f"SQL validation error: {str(e)}"

    def generate_sql(self, question: str, system_prompt: str = None) -> str:
        """Generate SQL using T5 Text-to-SQL model with enhanced context awareness"""
        try:
            # Always use T5 model - no fallbacks
            if self.tokenizer is None or self.text_to_sql_model is None:
                raise Exception("T5 model not loaded")

            # Enhanced T5 prompt with context awareness
            if system_prompt:
                # Use custom system prompt if provided by frontend
                input_text = f"{system_prompt}\n\nQuestion: {question}"
            else:
                # Context-aware prompt for better understanding
                input_text = f"""Context: US traffic accidents database with weather, location, and severity data.
Table: us_accidents
Columns: state, city, county, severity, start_time, end_time, weather_condition, temperature_f, wind_speed_mph, precipitation_in, visibility_mi, humidity_pct, pressure_in, wind_direction, description, street, zipcode, country, timezone, start_lat, start_lng, end_lat, end_lng, distance_mi, source, id, airport_code, weather_timestamp, wind_chill_f, amenity, bump, crossing, give_way, junction, no_exit, railway, roundabout, station, stop, traffic_calming, traffic_signal, turning_loop, sunrise_sunset, civil_twilight, nautical_twilight, astronomical_twilight

Generate PostgreSQL query for: {question}

SQL:"""

            logger.info(f"T5 input: {input_text[:200]}...")

            # Generate SQL using T5 model with optimized parameters
            inputs = self.tokenizer(
                input_text,
                return_tensors="pt",
                max_length=512,
                truncation=True,
                padding=True
            )

            with torch.no_grad():
                outputs = self.text_to_sql_model.generate(
                    inputs.input_ids,
                    max_length=100,
                    num_beams=3,
                    do_sample=False,   # Deterministic output to avoid nan/inf
                    early_stopping=True,
                    pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    no_repeat_ngram_size=2
                )

            generated_sql = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Clean the generated SQL
            generated_sql = generated_sql.strip()

            # Remove input prompt from output if present
            if "SQL:" in generated_sql:
                generated_sql = generated_sql.split("SQL:")[-1].strip()

            # Ensure it ends with semicolon
            if not generated_sql.endswith(';'):
                generated_sql += ';'

            logger.info(f"T5 model generated: {generated_sql}")
            return generated_sql

        except Exception as e:
            logger.error(f"Error generating SQL with T5 model: {str(e)}")
            # Return a generic error instead of fallback
            return "SELECT 'T5 model error - please try rephrasing your question' AS error;"


query_engine_instance = TextToSQLEngine()

@app.get("/")
async def root():
    return {"message": "OptimaX SQL Chat API is running"}

@app.get("/health")
async def health_check():
    try:
        with query_engine_instance.engine.connect() as conn:
            result = conn.execute(text("SELECT 1"))
            return {"status": "healthy", "database": "connected"}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}

@app.post("/chat", response_model=ChatResponse)
async def chat(message: ChatMessage):
    try:
        # Generate SQL using Text-to-SQL engine
        sql_query = query_engine_instance.generate_sql(
            question=message.message,
            system_prompt=message.system_prompt
        )

        logger.info(f"Generated SQL: {sql_query}")

        # Validate SQL for safety and performance
        is_valid, validated_sql = query_engine_instance.validate_sql(sql_query)
        if not is_valid:
            return ChatResponse(
                response=f"Query validation failed: {validated_sql}",
                sql_query=None,
                error=validated_sql
            )

        logger.info(f"Validated SQL: {validated_sql}")

        # Execute SQL with read-only approach
        with query_engine_instance.engine.connect() as conn:
            result = conn.execute(text(validated_sql))
            rows = result.fetchall()
            columns = result.keys()

        # Enhanced result formatting
        if rows:
            # Format as structured data for frontend
            formatted_data = []
            for i, row in enumerate(rows):
                if len(row) == 2 and isinstance(row[1], (int, float)):
                    # Key-value pair with numeric value
                    formatted_data.append(f"{i+1}. {row[0]} - {row[1]:,}")
                elif len(row) == 1:
                    # Single column result
                    formatted_data.append(f"{i+1}. {row[0]}")
                else:
                    # Multi-column result
                    row_data = []
                    for j, col in enumerate(row):
                        if isinstance(col, (int, float)) and col > 1000:
                            row_data.append(f"{col:,}")
                        else:
                            row_data.append(str(col))
                    formatted_data.append(f"{i+1}. {' | '.join(row_data)}")

            # Limit display to prevent overwhelming frontend
            if len(formatted_data) > 50:
                formatted_result = "\n".join(formatted_data[:50]) + f"\n\n... and {len(formatted_data) - 50} more results"
            else:
                formatted_result = "\n".join(formatted_data)
        else:
            formatted_result = "No results found."

        return ChatResponse(
            response=formatted_result,
            sql_query=validated_sql if os.getenv("DEBUG") == "true" else None
        )

    except Exception as e:
        logger.error(f"Error processing chat request: {str(e)}")
        return ChatResponse(
            response="Sorry, I encountered an error processing your request. Please try rephrasing your question.",
            sql_query=None,
            error=str(e)
        )

@app.get("/table-info")
async def get_table_info():
    try:
        with query_engine_instance.engine.connect() as conn:
            result = conn.execute(text("""
                SELECT column_name, data_type, is_nullable
                FROM information_schema.columns
                WHERE table_name = 'us_accidents'
                ORDER BY ordinal_position
            """))
            columns = [{"name": row[0], "type": row[1], "nullable": row[2]} for row in result]

            count_result = conn.execute(text("SELECT COUNT(*) FROM us_accidents"))
            total_records = count_result.scalar()

            return {
                "table": "us_accidents",
                "columns": columns,
                "total_records": total_records
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)