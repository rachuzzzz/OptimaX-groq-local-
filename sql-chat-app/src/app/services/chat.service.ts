import { Injectable } from '@angular/core';
import { HttpClient, HttpHeaders } from '@angular/common/http';
import { Observable } from 'rxjs';

export interface ChatMessage {
  message: string;
  system_prompt?: string;
  include_sql?: boolean;
  session_id?: string;
  include_tasks?: boolean;
  row_limit?: number;
}

export interface AgentTask {
  task_id: string;
  task_type: string;
  description: string;
  status: string;
  result?: any;
  error?: string;
  context?: any;
  dependencies?: string[];
}

export interface ChartSuggestion {
  analysis_type: 'comparison' | 'time_series' | 'proportion' | 'correlation' | 'distribution';
  recommended_charts: Array<{
    type: string;
    label: string;
    recommended: boolean;
  }>;
}

export interface ChatResponse {
  response: string;
  sql_query?: string;
  error?: string;
  query_results?: any[];
  session_id?: string;
  tasks?: AgentTask[];
  clarification_needed?: boolean;
  data?: any;
  execution_time?: number;
  agent_reasoning?: string;
  chart_suggestion?: ChartSuggestion;
}

export interface SessionInfo {
  session_id: string;
  created_at: string;
  last_active: string;
  message_count: number;
  query_count: number;
  task_count: number;
}

export interface TableInfo {
  table: string;
  columns: Array<{
    name: string;
    type: string;
    nullable: string;
  }>;
  total_records: number;
}

@Injectable({
  providedIn: 'root'
})
export class ChatService {
  // v4.0 backend (port 8000) - Single Groq LLM architecture
  private agenticMode = true;

  private get baseUrl(): string {
    return 'http://localhost:8000';  // v4.0 simplified backend
  }

  constructor(private http: HttpClient) {}

  // Toggle between agentic and optimized backends (kept for compatibility)
  setAgenticMode(enabled: boolean): void {
    this.agenticMode = enabled;
    console.log(`Mode: ${enabled ? 'agentic' : 'optimized'} (${this.baseUrl})`);
  }

  isAgenticMode(): boolean {
    return this.agenticMode;
  }

  sendMessage(message: ChatMessage): Observable<ChatResponse> {
    return this.http.post<ChatResponse>(`${this.baseUrl}/chat`, message, {
      headers: new HttpHeaders({
        'Content-Type': 'application/json'
      })
    });
  }

  getTableInfo(): Observable<TableInfo> {
    return this.http.get<TableInfo>(`${this.baseUrl}/table-info`);
  }

  checkHealth(): Observable<any> {
    return this.http.get(`${this.baseUrl}/health`);
  }

  // Session Management
  getSession(sessionId: string): Observable<SessionInfo> {
    return this.http.get<SessionInfo>(`${this.baseUrl}/sessions/${sessionId}`);
  }

  listSessions(): Observable<{ total_sessions: number; sessions: SessionInfo[] }> {
    return this.http.get<{ total_sessions: number; sessions: SessionInfo[] }>(`${this.baseUrl}/sessions`);
  }

  deleteSession(sessionId: string): Observable<{ message: string }> {
    return this.http.delete<{ message: string }>(`${this.baseUrl}/sessions/${sessionId}`);
  }

  // Performance metrics
  getPerformance(): Observable<any> {
    return this.http.get(`${this.baseUrl}/performance`);
  }

  // Agent info
  getAgentInfo(): Observable<any> {
    return this.http.get(`${this.baseUrl}/agent/info`);
  }

  // Database Management
  getDatabaseSchema(): Observable<any> {
    return this.http.get(`${this.baseUrl}/database/schema`);
  }

  testDatabaseConnection(databaseUrl: string): Observable<any> {
    return this.http.post(`${this.baseUrl}/database/test-connection`, {
      database_url: databaseUrl
    });
  }

  connectDatabase(databaseUrl: string): Observable<any> {
    return this.http.post(`${this.baseUrl}/database/connect`, {
      database_url: databaseUrl
    });
  }

  // System Prompt Management
  getActiveSystemPrompt(): Observable<any> {
    return this.http.get(`${this.baseUrl}/system-prompt/active`);
  }

  saveSystemPrompt(prompt: string, useDynamicSchema: boolean = true): Observable<any> {
    return this.http.post(`${this.baseUrl}/system-prompt/save`, {
      prompt: prompt,
      use_dynamic_schema: useDynamicSchema
    });
  }

  resetSystemPrompt(): Observable<any> {
    return this.http.post(`${this.baseUrl}/system-prompt/reset`, {});
  }

  applySystemPrompt(): Observable<any> {
    return this.http.post(`${this.baseUrl}/system-prompt/apply`, {});
  }
}