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

export interface ChartRecommendation {
  chart_type: 'bar' | 'line' | 'pie' | 'doughnut' | 'table' | 'none';
  reasoning: string;
  config?: {
    x_axis?: string;
    y_axis?: string;
    y_axes?: string[];
    labels?: string;
    values?: string;
    title?: string;
    columns?: string[];
  };
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
  chart_recommendation?: ChartRecommendation;
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
  // Use v4.0 backend by default (port 8000)
  // Keeping agenticMode flag for compatibility but always using v4.0
  private agenticMode = true;

  private get baseUrl(): string {
    return 'http://localhost:8000';  // v4.0 simplified backend
  }

  constructor(private http: HttpClient) {}

  // Toggle between agentic and optimized backends
  setAgenticMode(enabled: boolean): void {
    this.agenticMode = enabled;
    console.log(`Switched to ${enabled ? 'agentic' : 'optimized'} backend (${this.baseUrl})`);
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

  // Session Management (Agentic mode only)
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

  // Agent info (Agentic mode only)
  getAgentInfo(): Observable<any> {
    return this.http.get(`${this.baseUrl}/agent/info`);
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