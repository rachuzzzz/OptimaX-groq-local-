import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { trigger, state, style, transition, animate } from '@angular/animations';
import { ChatService, ChatResponse, AgentTask } from '../../services/chat.service';
import { SystemPromptManagerComponent } from '../system-prompt-manager/system-prompt-manager';
import { ChartVisualizationComponent, ChartData } from '../chart-visualization/chart-visualization';
import { ChartDetectionService } from '../../services/chart-detection.service';

interface Message {
  type: 'user' | 'ai' | 'thinking' | 'clarification';
  content: string;
  timestamp: Date;
  sqlQuery?: string;
  responseTime?: number;
  error?: boolean;
  chartData?: ChartData;
  showChart?: boolean;
  queryResults?: any[];
  tasks?: AgentTask[];
  clarificationNeeded?: boolean;
  agentReasoning?: string;
  sessionId?: string;
}

interface DebugLog {
  timestamp: Date;
  level: 'info' | 'warn' | 'error';
  message: string;
}

interface TableInfo {
  table: string;
  columns: Array<{name: string; type: string; nullable: string}>;
  total_records: number;
}

interface SQLQueryEntry {
  id: number;
  query: string;
  userQuestion: string;
  timestamp: Date;
  responseTime: number;
  success: boolean;
  error?: string;
}

@Component({
  selector: 'app-chat-interface',
  standalone: true,
  imports: [CommonModule, FormsModule, SystemPromptManagerComponent, ChartVisualizationComponent],
  templateUrl: './chat-interface.html',
  styleUrl: './chat-interface.scss',
  animations: [
    trigger('fadeInOut', [
      state('in', style({opacity: 1})),
      transition(':enter', [
        style({opacity: 0}),
        animate(300, style({opacity: 1}))
      ]),
      transition(':leave', [
        animate(300, style({opacity: 0}))
      ])
    ])
  ]
})
export class ChatInterfaceComponent implements OnInit {
  // System prompt - default prompt defined first
  private defaultSystemPrompt = `You are OptimaX, an expert SQL query generator and data analyst. Convert natural language questions into PostgreSQL queries and present results in a clear, formatted manner.

DATABASE SCHEMA - us_accidents table (7,728,394 records):
- id VARCHAR(10) PRIMARY KEY
- source VARCHAR(10)
- severity INTEGER (1=low, 2=minor, 3=major, 4=severe)
- start_time TIMESTAMP, end_time TIMESTAMP  
- start_lat DECIMAL(10,8), start_lng DECIMAL(11,8), end_lat DECIMAL(10,8), end_lng DECIMAL(11,8)
- distance_mi DECIMAL(8,3)
- description TEXT
- street VARCHAR(200), city VARCHAR(100), county VARCHAR(100), state VARCHAR(2), zipcode VARCHAR(10)
- country VARCHAR(2), timezone VARCHAR(50), airport_code VARCHAR(10)
- weather_timestamp TIMESTAMP
- temperature_f DECIMAL(8,2), wind_chill_f DECIMAL(8,2), humidity_pct DECIMAL(8,2)
- pressure_in DECIMAL(8,2), visibility_mi DECIMAL(8,2), wind_direction VARCHAR(10)
- wind_speed_mph DECIMAL(8,2), precipitation_in DECIMAL(8,3), weather_condition VARCHAR(100)
- amenity BOOLEAN, bump BOOLEAN, crossing BOOLEAN, give_way BOOLEAN, junction BOOLEAN
- no_exit BOOLEAN, railway BOOLEAN, roundabout BOOLEAN, station BOOLEAN, stop BOOLEAN
- traffic_calming BOOLEAN, traffic_signal BOOLEAN, turning_loop BOOLEAN
- sunrise_sunset VARCHAR(10), civil_twilight VARCHAR(10), nautical_twilight VARCHAR(10), astronomical_twilight VARCHAR(10)

INDEXES available: state, city, start_time, severity, (start_lat, start_lng)

RESPONSE FORMAT RULES:
1. Always show the actual SQL query used
2. Present results in a clear, numbered list format
3. Include exact numbers with comma separators (e.g., 1,741,433)
4. Use proper headings and formatting
5. Provide context and insights about the data

EXAMPLE RESPONSE FORMAT:
**SQL Query:**
\`\`\`sql
SELECT state, COUNT(*) as accident_count 
FROM us_accidents 
GROUP BY state 
ORDER BY accident_count DESC 
LIMIT 10;
\`\`\`

**Top 10 States with Most Accidents:**
1. **California (CA)** - 1,741,433 accidents
2. **Florida (FL)** - 880,192 accidents
3. **Texas (TX)** - 582,837 accidents
[continue numbered list...]

**Analysis:** California leads significantly with over 1.7 million recorded accidents...

SQL GENERATION RULES:
1. Generate ONLY valid PostgreSQL syntax
2. Use proper column names exactly as shown above
3. For location queries, use start_lat/start_lng coordinates
4. For time queries, use start_time column
5. Boolean columns: use TRUE/FALSE
6. Always include LIMIT for large result sets
7. Use appropriate WHERE clauses for performance

Example patterns:
- "accidents in California" → WHERE state = 'CA'
- "severe accidents" → WHERE severity = 4
- "accidents during snow" → WHERE weather_condition ILIKE '%snow%'
- "accidents near traffic signals" → WHERE traffic_signal = TRUE`;

  systemPrompt: string = this.defaultSystemPrompt;

  userInput: string = '';
  messages: Message[] = [];
  sidebarExpanded: boolean = true;
  isLoading: boolean = true; // Start with loading screen
  showSystemPromptManager: boolean = false;
  isUsingCustomPrompt: boolean = false;
  originalSystemPrompt: string = '';

  // Session management (Agentic mode)
  currentSessionId: string | null = null;
  sessionInfo: any = null;
  showSessionInfo: boolean = false;

  // Agentic mode toggle
  agenticMode: boolean = true;
  showAgenticFeatures: boolean = true;
  showTaskBreakdown: boolean = true;

  // Developer mode properties
  debugMode: boolean = false;
  showDebugPanel: boolean = false;
  activeDebugTab: 'logs' | 'performance' | 'sql' | 'sessions' = 'logs';
  debugLogs: DebugLog[] = [];

  // Backend connection and monitoring
  backendConnected: boolean = false;
  lastQueryTime: number | null = null;
  totalQueries: number = 0;
  avgResponseTime: number | null = null;
  responseTimes: number[] = [];
  lastGeneratedSQL: string | null = null;

  // Recent queries and suggestions
  recentQueries: string[] = [];
  autoCompleteEnabled: boolean = false;

  // Table information
  tableInfo: TableInfo | null = null;

  // SQL Query tracking for debugging
  sqlQueryHistory: SQLQueryEntry[] = [];
  sqlQueryIdCounter: number = 1;

  // Time-based greeting
  greeting: string = '';

  examplePrompts = [
    'Show me the top 10 states with most accidents',
    'Find severe accidents during snow weather',
    'What times of day have the most accidents?',
    'Show accidents near traffic signals in California',
    'Which weather conditions cause the most accidents?'
  ];

  constructor(
    private chatService: ChatService,
    private chartDetectionService: ChartDetectionService
  ) {
    this.loadFromStorage();
    this.setGreeting();

    // Set agentic mode in service
    this.chatService.setAgenticMode(this.agenticMode);
  }

  ngOnInit(): void {
    this.originalSystemPrompt = this.systemPrompt;
    this.addDebugLog('info', 'OptimaX Developer Console initialized');
    this.addDebugLog('info', `Running in ${this.agenticMode ? 'Agentic' : 'Optimized'} mode`);

    // Load or create session
    this.initializeSession();

    // Initialize the application with loading screen
    this.initializeApp();

    // Listen for test prompt events from system prompt manager
    document.addEventListener('testPrompt', (event: any) => {
      this.userInput = event.detail.message;
      // Auto-send the test message after a brief delay
      setTimeout(() => {
        this.sendMessage();
      }, 500);
    });

    // Listen for close system prompt manager events
    document.addEventListener('closeSystemPromptManager', () => {
      this.closeSystemPromptManager();
    });
  }

  private async initializeApp(): Promise<void> {
    try {
      // Simulate initialization steps
      await this.delay(1000); // Initial setup
      this.addDebugLog('info', 'Connecting to backend...');

      await this.delay(800);
      this.checkBackendConnection();
      this.addDebugLog('info', 'Backend connection established');

      await this.delay(500);
      this.addDebugLog('info', 'Loading system prompts...');

      await this.delay(700);
      this.addDebugLog('info', 'OptimaX ready for use');

      // Hide loading screen
      this.isLoading = false;
    } catch (error) {
      this.addDebugLog('error', 'Failed to initialize application');
      // Still hide loading screen even if there's an error
      setTimeout(() => {
        this.isLoading = false;
      }, 2000);
    }
  }

  private delay(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  sendMessage(): void {
    if (this.userInput.trim() && !this.isLoading) {
      const userMessage = this.userInput.trim();
      const queryStartTime = Date.now();

      // Add to recent queries
      this.recentQueries = this.recentQueries.filter(q => q !== userMessage);
      this.recentQueries.unshift(userMessage);
      this.recentQueries = this.recentQueries.slice(0, 10);

      this.messages.push({
        type: 'user',
        content: userMessage,
        timestamp: new Date()
      });

      // Add thinking indicator
      this.messages.push({
        type: 'thinking',
        content: 'OptimaX is thinking...',
        timestamp: new Date()
      });

      this.userInput = '';
      this.isLoading = true;
      this.totalQueries++;

      this.addDebugLog('info', `Sending query: "${userMessage.substring(0, 50)}..."`);

      const chatMessage = {
        message: userMessage,
        system_prompt: this.isUsingCustomPrompt ? this.systemPrompt : undefined,
        include_sql: true,
        session_id: this.currentSessionId || undefined,
        include_tasks: this.agenticMode && this.showTaskBreakdown,
        row_limit: 50
      };

      this.chatService.sendMessage(chatMessage).subscribe({
        next: (response: ChatResponse) => {
          const responseTime = Date.now() - queryStartTime;
          this.lastQueryTime = responseTime;

          // Store session ID
          if (response.session_id) {
            this.currentSessionId = response.session_id;
            this.saveSessionToStorage();
          }

          if (response.sql_query) {
            this.lastGeneratedSQL = response.sql_query;

            // Add to SQL query history for debugging
            this.sqlQueryHistory.unshift({
              id: this.sqlQueryIdCounter++,
              query: response.sql_query,
              userQuestion: userMessage,
              timestamp: new Date(),
              responseTime: responseTime,
              success: !response.error,
              error: response.error
            });

            // Keep only last 20 queries to prevent memory issues
            if (this.sqlQueryHistory.length > 20) {
              this.sqlQueryHistory = this.sqlQueryHistory.slice(0, 20);
            }
          }

          // Remove thinking indicator
          this.messages = this.messages.filter(msg => msg.type !== 'thinking');

          // Detect if results should be visualized
          let chartData: ChartData | undefined;
          let showChart = false;

          console.log('=== CHART DETECTION DEBUG ===');
          console.log('Has sql_query:', !!response.sql_query);
          console.log('Has query_results:', !!response.query_results);
          console.log('Has chart_recommendation:', !!response.chart_recommendation);
          console.log('Query results length:', response.query_results?.length);
          console.log('Chart recommendation:', response.chart_recommendation);

          // Primary: Use backend chart recommendation if available
          if (response.chart_recommendation &&
              response.chart_recommendation.chart_type &&
              response.chart_recommendation.chart_type !== 'none' &&
              response.chart_recommendation.chart_type !== 'table' &&
              response.query_results &&
              response.query_results.length > 0) {

            const chartType = response.chart_recommendation.chart_type as any;
            console.log('Using BACKEND chart recommendation:', chartType);

            chartData = this.chartDetectionService.parseToChartData(
              response.query_results,
              chartType
            );
            console.log('Chart data from backend recommendation:', chartData);

            showChart = true;
            this.addDebugLog('info', `Backend recommended chart: ${chartType} - ${response.chart_recommendation.reasoning}`);
          }
          // Fallback: Use frontend detection if user explicitly requested visualization
          else if (response.sql_query && response.query_results && response.query_results.length > 0) {
            const shouldViz = this.chartDetectionService.shouldVisualize(
              response.sql_query,
              response.query_results,
              userMessage  // Pass user message to check for visualization intent
            );

            console.log('Fallback: Frontend should visualize:', shouldViz);

            if (shouldViz) {
              const chartType = this.chartDetectionService.detectChartType(
                response.sql_query,
                response.query_results
              );
              console.log('Fallback: Chart type detected:', chartType);

              chartData = this.chartDetectionService.parseToChartData(
                response.query_results,
                chartType
              );
              console.log('Fallback: Chart data:', chartData);

              showChart = true;
              this.addDebugLog('info', `Frontend fallback chart: ${chartType} with ${response.query_results.length} data points`);
            }
          } else {
            console.log('Chart detection skipped - no recommendation and missing data');
          }

          // Handle clarification needed
          const messageType: 'ai' | 'clarification' = response.clarification_needed ? 'clarification' : 'ai';

          this.messages.push({
            type: messageType,
            content: response.response,
            timestamp: new Date(),
            sqlQuery: response.sql_query,
            responseTime: responseTime,
            error: !!response.error,
            chartData: chartData,
            showChart: showChart,
            queryResults: response.query_results,
            tasks: response.tasks,
            clarificationNeeded: response.clarification_needed,
            agentReasoning: response.agent_reasoning,
            sessionId: response.session_id
          });

          this.addDebugLog('info', `Query completed in ${responseTime}ms`);
          if (response.error) {
            this.addDebugLog('error', `Query error: ${response.error}`);
          }

          this.isLoading = false;
          this.saveToStorage();
        },
        error: (error) => {
          const responseTime = Date.now() - queryStartTime;
          this.addDebugLog('error', `Query failed: ${error.message || 'Unknown error'}`);

          // Remove thinking indicator
          this.messages = this.messages.filter(msg => msg.type !== 'thinking');

          this.messages.push({
            type: 'ai',
            content: 'Sorry, I encountered an error processing your request. Please make sure the backend server is running.',
            timestamp: new Date(),
            error: true,
            responseTime: responseTime
          });
          this.isLoading = false;
          this.saveToStorage();
          console.error('Chat error:', error);
        }
      });
    }
  }

  clearChat(): void {
    this.messages = [];
  }

  onKeyPress(event: KeyboardEvent): void {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      this.sendMessage();
    }
  }

  useExamplePrompt(prompt: string): void {
    this.userInput = prompt;
  }

  toggleSidebar(): void {
    this.sidebarExpanded = !this.sidebarExpanded;
  }


  openSystemPromptManager(): void {
    this.showSystemPromptManager = true;
  }

  closeSystemPromptManager(): void {
    this.showSystemPromptManager = false;
  }


  getPromptStats(): string {
    const chars = this.systemPrompt.length;
    const words = this.systemPrompt.trim().split(/\s+/).length;
    const lines = this.systemPrompt.split('\n').length;
    return `${chars} characters, ${words} words, ${lines} lines`;
  }

  // Developer mode functions
  toggleDebugMode(): void {
    this.debugMode = !this.debugMode;
    this.addDebugLog('info', `Debug mode ${this.debugMode ? 'enabled' : 'disabled'}`);
    this.saveToStorage();
  }


  toggleDebugPanel(): void {
    this.showDebugPanel = !this.showDebugPanel;
  }

  toggleAutoComplete(): void {
    this.autoCompleteEnabled = !this.autoCompleteEnabled;
    this.addDebugLog('info', `Auto-complete ${this.autoCompleteEnabled ? 'enabled' : 'disabled'}`);
  }

  // Backend connection monitoring
  checkBackendConnection(): void {
    this.chatService.checkHealth().subscribe({
      next: (response) => {
        this.backendConnected = true;
      },
      error: (error) => {
        this.backendConnected = false;
      }
    });
  }

  // Debug logging
  addDebugLog(level: 'info' | 'warn' | 'error', message: string): void {
    this.debugLogs.push({
      timestamp: new Date(),
      level,
      message
    });

    if (this.debugLogs.length > 100) {
      this.debugLogs = this.debugLogs.slice(-100);
    }
  }

  // Storage functions
  saveToStorage(): void {
    const data = {
      recentQueries: this.recentQueries,
      debugMode: this.debugMode,
      debugLogs: this.debugLogs.slice(-20)
    };
    localStorage.setItem('optimax-developer-data', JSON.stringify(data));
  }

  loadFromStorage(): void {
    const stored = localStorage.getItem('optimax-developer-data');
    if (stored) {
      try {
        const data = JSON.parse(stored);
        this.recentQueries = data.recentQueries || [];
        this.debugMode = data.debugMode || false;
        this.debugLogs = data.debugLogs || [];
      } catch (e) {
        console.error('Failed to load saved data');
      }
    }
  }

  // Additional methods for new features
  useRecentQuery(query: string): void {
    this.userInput = query;
  }

  showTableInfo(): void {
    this.chatService.getTableInfo().subscribe({
      next: (info) => {
        alert(`Table: ${info.table}\nColumns: ${info.columns.length}\nRecords: ${info.total_records.toLocaleString()}`);
      },
      error: (error) => {
        alert('Failed to load table information. Please check backend connection.');
      }
    });
  }

  showConnectionStatus(): void {
    const status = this.backendConnected ? 'Connected' : 'Disconnected';
    alert(`Backend Status: ${status}`);
  }

  copyToClipboard(content: string): void {
    navigator.clipboard.writeText(content);
  }

  regenerateResponse(messageIndex: number): void {
    if (messageIndex > 0) {
      const userMessage = this.messages[messageIndex - 1];
      this.userInput = userMessage.content;
    }
  }

  showSqlDetails(sql: string): void {
    alert(`Generated SQL:\n\n${sql}`);
  }

  exportHistory(): void {
    const dataStr = JSON.stringify(this.messages, null, 2);
    const dataBlob = new Blob([dataStr], {type: 'application/json'});
    const link = document.createElement('a');
    link.href = URL.createObjectURL(dataBlob);
    link.download = `optimax-chat-export-${new Date().toISOString().split('T')[0]}.json`;
    link.click();
  }

  showQueryAnalytics(): void {
    alert(`Query Analytics:\n\nTotal Queries: ${this.totalQueries}\nMessages: ${this.messages.length}`);
  }


  onInputChange(): void {
    // Auto-complete logic placeholder
  }

  // SQL Debug utility functions
  clearSQLHistory(): void {
    this.sqlQueryHistory = [];
    this.addDebugLog('info', 'SQL query history cleared');
  }

  copySQLQuery(query: string): void {
    navigator.clipboard.writeText(query);
    this.addDebugLog('info', 'SQL query copied to clipboard');
  }

  formatSQLQuery(query: string): string {
    // Basic SQL formatting
    return query
      .replace(/SELECT/gi, '\nSELECT')
      .replace(/FROM/gi, '\nFROM')
      .replace(/WHERE/gi, '\nWHERE')
      .replace(/GROUP BY/gi, '\nGROUP BY')
      .replace(/ORDER BY/gi, '\nORDER BY')
      .replace(/HAVING/gi, '\nHAVING')
      .replace(/LIMIT/gi, '\nLIMIT')
      .trim();
  }

  getSQLStats(): { total: number; successful: number; failed: number; avgResponseTime: number } {
    const total = this.sqlQueryHistory.length;
    const successful = this.sqlQueryHistory.filter(q => q.success).length;
    const failed = total - successful;
    const avgResponseTime = total > 0
      ? Math.round(this.sqlQueryHistory.reduce((sum, q) => sum + q.responseTime, 0) / total)
      : 0;

    return { total, successful, failed, avgResponseTime };
  }

  setGreeting(): void {
    const hour = new Date().getHours();

    if (hour >= 5 && hour < 12) {
      this.greeting = 'Good Morning';
    } else if (hour >= 12 && hour < 17) {
      this.greeting = 'Good Afternoon';
    } else if (hour >= 17 && hour < 22) {
      this.greeting = 'Good Evening';
    } else {
      this.greeting = 'Good Night';
    }
  }

  // Table rendering helper functions
  getTableColumns(results: any[]): string[] {
    if (!results || results.length === 0) return [];
    return Object.keys(results[0]);
  }

  formatCellValue(value: any): string {
    if (value === null || value === undefined) {
      return 'NULL';
    }
    if (typeof value === 'number') {
      // Format large numbers with commas
      if (value > 1000 || value < -1000) {
        return value.toLocaleString();
      }
      return value.toString();
    }
    if (typeof value === 'boolean') {
      return value ? 'true' : 'false';
    }
    return String(value);
  }

  // ========================================================================
  // Agentic Mode Methods
  // ========================================================================

  toggleAgenticMode(): void {
    this.agenticMode = !this.agenticMode;
    this.chatService.setAgenticMode(this.agenticMode);
    this.addDebugLog('info', `Switched to ${this.agenticMode ? 'Agentic' : 'Optimized'} mode`);

    if (!this.agenticMode) {
      // Clear session when switching to optimized mode
      this.currentSessionId = null;
    } else {
      // Create new session when switching to agentic mode
      this.initializeSession();
    }

    this.saveToStorage();
  }

  toggleTaskBreakdown(): void {
    this.showTaskBreakdown = !this.showTaskBreakdown;
    this.addDebugLog('info', `Task breakdown ${this.showTaskBreakdown ? 'enabled' : 'disabled'}`);
  }

  toggleAgenticFeatures(): void {
    this.showAgenticFeatures = !this.showAgenticFeatures;
  }

  // Session Management
  initializeSession(): void {
    if (!this.agenticMode) return;

    // Try to load session from storage
    const storedSessionId = localStorage.getItem('optimax-session-id');

    if (storedSessionId) {
      this.currentSessionId = storedSessionId;
      this.addDebugLog('info', `Loaded session: ${storedSessionId.substring(0, 8)}...`);
    } else {
      // New session will be created on first message
      this.addDebugLog('info', 'New session will be created on first query');
    }
  }

  saveSessionToStorage(): void {
    if (this.currentSessionId) {
      localStorage.setItem('optimax-session-id', this.currentSessionId);
    }
  }

  clearSession(): void {
    if (this.currentSessionId && this.agenticMode) {
      this.chatService.deleteSession(this.currentSessionId).subscribe({
        next: (response) => {
          this.addDebugLog('info', `Session cleared: ${this.currentSessionId}`);
          this.currentSessionId = null;
          localStorage.removeItem('optimax-session-id');
          this.clearChat();
        },
        error: (error) => {
          this.addDebugLog('error', `Failed to clear session: ${error.message}`);
        }
      });
    } else {
      this.clearChat();
    }
  }

  viewSessionInfo(): void {
    if (this.currentSessionId && this.agenticMode) {
      this.chatService.getSession(this.currentSessionId).subscribe({
        next: (info) => {
          this.sessionInfo = info;
          this.showSessionInfo = true;
          this.addDebugLog('info', 'Session info loaded');
        },
        error: (error) => {
          this.addDebugLog('error', `Failed to load session info: ${error.message}`);
          alert('Failed to load session information');
        }
      });
    } else {
      alert('No active session or not in agentic mode');
    }
  }

  closeSessionInfo(): void {
    this.showSessionInfo = false;
  }

  // Task Methods
  getTaskIcon(taskType: string): string {
    const icons: { [key: string]: string } = {
      'sql_query': '🔍',
      'sql_analysis': '📊',
      'clarification': '❓',
      'multi_query': '🔗',
      'chat': '💬',
      'error': '❌'
    };
    return icons[taskType] || '📌';
  }

  getTaskStatusClass(status: string): string {
    const classes: { [key: string]: string } = {
      'pending': 'status-pending',
      'in_progress': 'status-inprogress',
      'completed': 'status-completed',
      'blocked': 'status-blocked',
      'failed': 'status-failed'
    };
    return classes[status] || '';
  }

  getTaskTypeName(taskType: string): string {
    const names: { [key: string]: string } = {
      'sql_query': 'SQL Query',
      'sql_analysis': 'Analysis',
      'clarification': 'Clarification',
      'multi_query': 'Multi-Query',
      'chat': 'Chat',
      'error': 'Error'
    };
    return names[taskType] || taskType;
  }
}