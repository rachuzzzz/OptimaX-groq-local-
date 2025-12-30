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
  chartSuggestion?: any;  // LLM suggested chart types
  awaitingChartChoice?: boolean;  // Waiting for user to pick chart type
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
  private defaultSystemPrompt = `You are OptimaX, an expert SQL query generator and data analyst. Convert natural language questions into PostgreSQL queries for airline data and present results in a clear, formatted manner.

DATABASE SCHEMA - postgres_air (airline booking and flight data):

TABLES:
- postgres_air.flight: flight_id, flight_no, scheduled_departure, scheduled_arrival, departure_airport, arrival_airport, status, aircraft_code, actual_departure, actual_arrival, update_ts
- postgres_air.booking: booking_id, booking_ref, booking_name, account_id, email, phone, price, update_ts
- postgres_air.passenger: passenger_id, booking_id, first_name, last_name, age, account_id, update_ts
- postgres_air.airport: airport_code, airport_name, city, airport_tz, continent, iso_country, iso_region, intnl, update_ts
- postgres_air.aircraft: code, model, range, class, velocity
- postgres_air.boarding_pass: pass_id, passenger_id, booking_leg_id, seat, boarding_time, precheck, update_ts
- postgres_air.booking_leg: booking_leg_id, booking_id, flight_id, leg_num, is_returning, update_ts
- postgres_air.account: account_id, login, first_name, last_name, update_ts
- postgres_air.phone: phone_id, account_id, phone, phone_type, primary_phone, update_ts
- postgres_air.frequent_flyer: frequent_flyer_id, account_id, airline, level, update_ts

KEY FIELDS:
- Airport codes: 3-letter codes (JFK, LAX, ORD, etc.)
- Timestamps: scheduled_departure, scheduled_arrival, actual_departure, actual_arrival (with time zone)
- Status: scheduled, departed, arrived, cancelled, delayed
- Price: numeric(7,2)
- Schema: ALWAYS prefix with postgres_air.

RESPONSE FORMAT RULES:
1. Always show the actual SQL query used
2. Present results in a clear, numbered list format
3. Include exact numbers with comma separators
4. Use proper headings and formatting
5. Provide context and insights about the data

EXAMPLE RESPONSE FORMAT:
**SQL Query:**
\`\`\`sql
SELECT departure_airport, arrival_airport, COUNT(*) as flight_count
FROM postgres_air.flight
GROUP BY departure_airport, arrival_airport
ORDER BY flight_count DESC
LIMIT 10;
\`\`\`

**Top 10 Routes by Flight Count:**
1. **JFK → LAX** - 1,234 flights
2. **LAX → JFK** - 1,198 flights
[continue numbered list...]

**Analysis:** The JFK-LAX route is the busiest with over 1,200 flights in each direction...

SQL GENERATION RULES:
1. Generate ONLY valid PostgreSQL syntax
2. ALWAYS use schema prefix: postgres_air.table_name
3. Use proper column names exactly as shown above
4. For time queries, use scheduled_departure, scheduled_arrival
5. Always include LIMIT for large result sets
6. Use appropriate WHERE clauses for performance

Example patterns:
- "flights from JFK" → WHERE departure_airport = 'JFK'
- "delayed flights" → WHERE status = 'delayed'
- "bookings in December" → WHERE EXTRACT(MONTH FROM update_ts) = 12
- "average booking price" → SELECT AVG(price) FROM postgres_air.booking`;

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
  tableCount: number = 0;

  // Database settings
  showDatabaseSettings: boolean = false;
  databaseSchema: any = null;
  expandedTables: { [key: string]: boolean } = {};
  testConnectionUrl: string = '';
  isTestingConnection: boolean = false;
  connectionTestResult: any = null;

  // New database connection
  newDatabaseUrl: string = '';
  isConnectingDatabase: boolean = false;
  newConnectionResult: any = null;

  // SQL Query tracking for debugging
  sqlQueryHistory: SQLQueryEntry[] = [];
  sqlQueryIdCounter: number = 1;

  // Time-based greeting
  greeting: string = '';

  // Cache for last chart recommendation (for zero-overhead visualize commands)
  lastChartRecommendation: {
    chartData?: ChartData;
    queryResults?: any[];
    sqlQuery?: string;
    timestamp: Date;
    chartSuggestion?: any;
  } | null = null;

  examplePrompts = [
    'Show me the top 10 busiest flight routes',
    'What are the most popular departure airports?',
    'Find all flights from JFK to LAX',
    'Show me the average booking price',
    'Which airports have international flights?',
    'How many passengers are there in the database?'
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
    // Load custom system prompt from localStorage if available
    this.loadSystemPromptFromStorage();

    this.originalSystemPrompt = this.systemPrompt;
    this.addDebugLog('info', 'OptimaX Developer Console initialized');
    this.addDebugLog('info', `Running in ${this.agenticMode ? 'Agentic' : 'Optimized'} mode`);

    if (this.isUsingCustomPrompt) {
      this.addDebugLog('info', 'Using custom system prompt');
    }

    // Load or create session
    this.initializeSession();

    // Initialize the application with loading screen
    this.initializeApp();

    // Load database schema
    this.loadDatabaseSchema();

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

    // Listen for prompt changes from system prompt manager
    document.addEventListener('systemPromptChanged', (event: any) => {
      this.loadSystemPromptFromStorage();
      this.addDebugLog('info', 'System prompt updated');
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

      // Check if this is a visualization command
      // NOTE: We ONLY instant-replay if we already have an LLM-generated chart
      // Otherwise, always go to backend to let LLM reason about chart type
      const visualizeKeywords = ['visualize', 'vizualize', 'show as chart', 'plot', 'show chart', 'display chart', 'pictorially'];
      const isVisualizeCommand = visualizeKeywords.some(keyword =>
        userMessage.toLowerCase().includes(keyword)
      );

      // If user wants to visualize and we ALREADY have an LLM-generated chart cached, replay it instantly
      if (isVisualizeCommand && this.lastChartRecommendation && this.lastChartRecommendation.chartData) {
        this.addDebugLog('info', '⚡ Instant replay of LLM-generated chart (zero backend calls)');

        // Add user message
        this.messages.push({
          type: 'user',
          content: userMessage,
          timestamp: new Date()
        });

        // Add AI response with cached LLM chart
        this.messages.push({
          type: 'ai',
          content: `Here's the visualization from my previous analysis:`,
          timestamp: new Date(),
          chartData: this.lastChartRecommendation.chartData,
          showChart: true,
          queryResults: this.lastChartRecommendation.queryResults,
          sqlQuery: this.lastChartRecommendation.sqlQuery,
          responseTime: 0 // Instant!
        });

        this.userInput = '';
        return; // Skip backend call - chart already generated by LLM
      }

      // If user asks for visualization but no chart cached, ALWAYS go to backend
      // Let the LLM reason about chart type from scratch
      if (isVisualizeCommand) {
        this.addDebugLog('info', '→ Sending to LLM for chart reasoning (pure AI decision)');
        // Continue to backend call below - LLM will analyze data and choose chart type
      }

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

          // LLM CHART SUGGESTION SYSTEM - User chooses chart type
          let chartData: ChartData | undefined;
          let showChart = false;
          let chartSuggestion: any = undefined;
          let awaitingChartChoice = false;

          // Check if LLM provided chart suggestions (not rendered yet - waiting for user choice)
          if (response.chart_suggestion &&
              response.chart_suggestion.recommended_charts &&
              response.chart_suggestion.recommended_charts.length > 0) {

            this.addDebugLog('info', `✓ LLM suggested chart types: ${response.chart_suggestion.analysis_type}`);

            // Store suggestions for user to choose from
            chartSuggestion = response.chart_suggestion;
            awaitingChartChoice = true;

            // Cache the data and suggestion for when user picks
            this.lastChartRecommendation = {
              chartData: undefined, // No chart rendered yet
              queryResults: response.query_results,
              sqlQuery: response.sql_query,
              timestamp: new Date(),
              chartSuggestion: chartSuggestion
            };

            this.addDebugLog('info', `Awaiting user choice from ${chartSuggestion.recommended_charts.length} options`);

          } else if (response.query_results && response.query_results.length > 0) {
            // Data returned but no chart suggestion - cache for potential future visualization
            this.lastChartRecommendation = {
              chartData: undefined,
              queryResults: response.query_results,
              sqlQuery: response.sql_query,
              timestamp: new Date()
            };
            this.addDebugLog('info', `Cached ${response.query_results.length} rows - ready for visualization`);
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
            sessionId: response.session_id,
            chartSuggestion: chartSuggestion,
            awaitingChartChoice: awaitingChartChoice
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
    // Clear the session to start fresh with updated system prompt
    this.currentSessionId = null;
    localStorage.removeItem('optimax-session-id');
    this.addDebugLog('info', 'Chat cleared - next message will create a new session');
    if (this.isUsingCustomPrompt) {
      this.addDebugLog('info', 'New session will use the custom system prompt');
    }
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
    // Reload system prompt after closing in case it was changed
    this.loadSystemPromptFromStorage();
  }

  loadSystemPromptFromStorage(): void {
    const savedPrompt = localStorage.getItem('optimax-system-prompt');
    if (savedPrompt) {
      this.systemPrompt = savedPrompt;
      this.isUsingCustomPrompt = true;
      this.addDebugLog('info', 'Loaded custom system prompt from storage');
    } else {
      this.systemPrompt = this.defaultSystemPrompt;
      this.isUsingCustomPrompt = false;
    }
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

  // Handle user selecting a chart type from suggestions
  selectChartType(chartType: string, messageIndex: number): void {
    if (!this.lastChartRecommendation || !this.lastChartRecommendation.queryResults) {
      this.addDebugLog('error', 'No cached data found for chart generation');
      return;
    }

    this.addDebugLog('info', `User selected chart type: ${chartType}`);

    // Generate chart data using selected type
    const chartData = this.chartDetectionService.parseToChartData(
      this.lastChartRecommendation.queryResults,
      chartType as any
    );

    // Add new message showing the selected chart
    this.messages.push({
      type: 'ai',
      content: `Here's your ${chartType.replace('_', ' ')} visualization:`,
      timestamp: new Date(),
      chartData: chartData,
      showChart: true,
      queryResults: this.lastChartRecommendation.queryResults,
      sqlQuery: this.lastChartRecommendation.sqlQuery,
      responseTime: 0 // Instant render from cached data
    });

    // Update cache with generated chart for instant replay
    this.lastChartRecommendation.chartData = chartData;

    // Mark the original message as no longer awaiting choice
    if (messageIndex >= 0 && messageIndex < this.messages.length) {
      this.messages[messageIndex].awaitingChartChoice = false;
    }

    this.addDebugLog('info', `Rendered ${chartType} chart from cached data`);
  }

  // Database Management Methods
  openDatabaseSettings(): void {
    this.showDatabaseSettings = true;
    this.loadDatabaseSchema();
  }

  closeDatabaseSettings(): void {
    this.showDatabaseSettings = false;
    this.connectionTestResult = null;
  }

  loadDatabaseSchema(): void {
    this.chatService.getDatabaseSchema().subscribe({
      next: (response: any) => {
        this.databaseSchema = response.schema;
        this.tableCount = response.schema.table_count || 0;
        this.addDebugLog('info', `Loaded schema: ${this.tableCount} tables`);
      },
      error: (error) => {
        this.addDebugLog('error', `Failed to load schema: ${error.message}`);
      }
    });
  }

  toggleTableExpand(tableName: string): void {
    this.expandedTables[tableName] = !this.expandedTables[tableName];
  }

  testConnection(): void {
    if (!this.testConnectionUrl.trim()) {
      alert('Please enter a database connection URL');
      return;
    }

    this.isTestingConnection = true;
    this.connectionTestResult = null;

    this.chatService.testDatabaseConnection(this.testConnectionUrl).subscribe({
      next: (result: any) => {
        this.connectionTestResult = result;
        this.isTestingConnection = false;

        if (result.success) {
          this.addDebugLog('info', `Connection test successful: ${result.table_count} tables found`);
        } else {
          this.addDebugLog('error', `Connection test failed: ${result.error}`);
        }
      },
      error: (error) => {
        this.connectionTestResult = {
          success: false,
          error: error.message || 'Connection test failed'
        };
        this.isTestingConnection = false;
        this.addDebugLog('error', `Connection test failed: ${error.message}`);
      }
    });
  }

  connectToNewDatabase(): void {
    if (!this.newDatabaseUrl.trim()) {
      alert('Please enter a database connection URL');
      return;
    }

    this.isConnectingDatabase = true;
    this.newConnectionResult = null;

    this.chatService.connectDatabase(this.newDatabaseUrl).subscribe({
      next: (result: any) => {
        this.newConnectionResult = result;
        this.isConnectingDatabase = false;

        if (result.success) {
          this.addDebugLog('info', `Connected to database: ${result.table_count} tables loaded`);
          // Reload schema to show the new database
          this.loadDatabaseSchema();
          // Update table count
          this.tableCount = result.table_count;
        } else {
          this.addDebugLog('error', `Connection failed: ${result.error}`);
        }
      },
      error: (error) => {
        this.newConnectionResult = {
          success: false,
          error: error.message || 'Connection failed'
        };
        this.isConnectingDatabase = false;
        this.addDebugLog('error', `Connection failed: ${error.message}`);
      }
    });
  }

  fillExample(exampleUrl: string): void {
    this.newDatabaseUrl = exampleUrl;
    this.newConnectionResult = null;
  }
}