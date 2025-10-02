import { Injectable } from '@angular/core';
import { HttpClient, HttpHeaders } from '@angular/common/http';
import { Observable } from 'rxjs';

export interface ChatMessage {
  message: string;
  system_prompt?: string;
  include_sql?: boolean;
}

export interface ChatResponse {
  response: string;
  sql_query?: string;
  error?: string;
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

export interface SystemPromptsResponse {
  intent_prompt: string;
  sql_prompt: string;
  default_intent_prompt: string;
  default_sql_prompt: string;
}

export interface SystemPromptUpdate {
  model_type: 'intent' | 'sql';
  prompt: string;
}

@Injectable({
  providedIn: 'root'
})
export class ChatService {
  private baseUrl = 'http://localhost:8002';

  constructor(private http: HttpClient) {}

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

  // System Prompt Management Methods
  getSystemPrompts(): Observable<SystemPromptsResponse> {
    return this.http.get<SystemPromptsResponse>(`${this.baseUrl}/system-prompts`);
  }

  updateSystemPrompt(promptUpdate: SystemPromptUpdate): Observable<any> {
    return this.http.post(`${this.baseUrl}/system-prompts`, promptUpdate, {
      headers: new HttpHeaders({
        'Content-Type': 'application/json'
      })
    });
  }

  resetAllSystemPrompts(): Observable<any> {
    return this.http.post(`${this.baseUrl}/system-prompts/reset`, {});
  }

  resetSystemPrompt(modelType: 'intent' | 'sql'): Observable<any> {
    return this.http.post(`${this.baseUrl}/system-prompts/reset/${modelType}`, {});
  }
}