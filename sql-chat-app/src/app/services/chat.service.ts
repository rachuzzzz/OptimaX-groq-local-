import { Injectable } from '@angular/core';
import { HttpClient, HttpHeaders } from '@angular/common/http';
import { Observable } from 'rxjs';

export interface ChatMessage {
  message: string;
  system_prompt?: string;
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

@Injectable({
  providedIn: 'root'
})
export class ChatService {
  private baseUrl = 'http://localhost:8001';

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
}