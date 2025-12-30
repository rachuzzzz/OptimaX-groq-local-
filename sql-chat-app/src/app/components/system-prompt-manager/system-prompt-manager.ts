import { Component, OnInit, Output, EventEmitter, inject } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { ChatService } from '../../services/chat.service';

@Component({
  selector: 'app-system-prompt-manager',
  standalone: true,
  imports: [CommonModule, FormsModule],
  template: `
    <div class="system-prompt-manager">
      <div class="manager-header">
        <h2>System Prompt Editor</h2>
        <p>Configure OptimaX v4.2 - Single Groq LLM (llama-3.3-70b-versatile)</p>
        <div class="architecture-info">
          <span class="badge">🚦 4-Gate Intent Routing</span>
          <span class="badge">⚡ Fast Response</span>
          <span class="badge">🎯 Single LLM</span>
        </div>
      </div>

      <div *ngIf="message" class="message" [ngClass]="'message-' + messageType">
        {{ message }}
      </div>

      <div class="editor">
        <div class="editor-header">
          <h3>Master System Prompt</h3>
          <div class="stats">
            <span>{{ promptStats.characters }} chars</span>
            <span>{{ promptStats.words }} words</span>
            <span>{{ promptStats.lines }} lines</span>
          </div>
        </div>
        <p class="description">
          Controls how OptimaX v4.2 handles SQL queries through 4-gate intent routing and ReActAgent execution.
        </p>

        <div class="warning-box" *ngIf="isDifferentFromDefault()">
          ⚠️ Custom prompt active - Using modified system prompt
        </div>

        <div class="workflow-info">
          <strong>Workflow:</strong> 1️⃣ Edit → 2️⃣ Save → 3️⃣ Apply Changes
        </div>

        <textarea
          class="prompt-input"
          [(ngModel)]="systemPrompt"
          (ngModelChange)="onPromptChange()"
          placeholder="Loading current system prompt..."
          spellcheck="false">
        </textarea>

        <div class="prompt-guidelines">
          <details>
            <summary>💡 Prompt Guidelines</summary>
            <ul>
              <li>Use {{ '{SCHEMA_SECTION}' }} placeholder for dynamic schema injection</li>
              <li>Include EFFICIENCY RULES to prevent excessive queries</li>
              <li>Specify when to use execute_sql vs get_schema tools</li>
              <li>Define STOP conditions to prevent max iterations errors</li>
              <li>Add safety rules (read-only queries, LIMIT enforcement)</li>
            </ul>
          </details>
        </div>

        <div class="actions">
          <button class="btn primary" (click)="savePrompt()" [disabled]="saveInProgress || !hasChanges()">
            {{ saveInProgress ? 'Saving...' : '💾 Save Prompt' }}
          </button>
          <button class="btn apply" (click)="applyPrompt()" [disabled]="applyInProgress">
            {{ applyInProgress ? 'Applying...' : '⚡ Apply Changes' }}
          </button>
          <button class="btn secondary" (click)="resetPrompt()" [disabled]="saveInProgress || applyInProgress">
            🔄 Reset to Default
          </button>
          <button class="btn secondary" (click)="exportPrompt()">📥 Export</button>
          <button class="btn secondary" (click)="testPrompt()">🧪 Test</button>
        </div>
      </div>

      <div class="info-panel">
        <h4>ℹ️ How It Works</h4>
        <p>OptimaX v4.2 uses <strong>4-gate intent routing</strong> with a single LLM (Groq llama-3.3-70b-versatile):</p>
        <ul>
          <li><strong>Gate 1: Visualization</strong> - One-shot chart classification (no agent)</li>
          <li><strong>Gate 2: Greeting</strong> - Fast-path response (&lt;50ms)</li>
          <li><strong>Gate 3: Ambiguous Entity</strong> - Disambiguation for unclear queries</li>
          <li><strong>Gate 4: Database Action</strong> - SQL queries via ReActAgent (max 5 iterations)</li>
        </ul>
        <div class="workflow-steps">
          <h5>📝 Workflow:</h5>
          <ol>
            <li><strong>Edit</strong> - Modify the system prompt below</li>
            <li><strong>Save</strong> - Saves to backend permanently (file: custom_system_prompt.txt)</li>
            <li><strong>Apply Changes</strong> - Hot-reloads agent without restart (clears sessions)</li>
            <li><strong>Test</strong> - Try it out with sample queries</li>
          </ol>
          <p class="tip">💡 <strong>Tip:</strong> Use <code>{{ '{SCHEMA_SECTION}' }}</code> in your prompt for dynamic schema injection.</p>
        </div>
      </div>

      <!-- Active Prompt Viewer -->
      <div class="active-prompt-viewer" *ngIf="showActivePrompt && activePromptInfo">
        <h4>🔍 Currently Active Prompt</h4>
        <div class="prompt-type-badge" [class.custom]="activePromptInfo.type === 'custom'">
          {{ activePromptInfo.type === 'custom' ? 'CUSTOM' : 'DEFAULT' }} PROMPT
        </div>
        <div class="active-prompt-content">
          <pre>{{ activePromptInfo.prompt }}</pre>
        </div>
        <button class="btn secondary" (click)="closeActivePrompt()">Close</button>
      </div>
    </div>
  `,
  styleUrl: './system-prompt-manager.scss'
})
export class SystemPromptManagerComponent implements OnInit {
  @Output() promptChanged = new EventEmitter<string>();

  private chatService = inject(ChatService);

  // Current prompt
  systemPrompt: string = '';
  useDynamicSchema: boolean = true;
  activePromptInfo: any = null;
  showActivePrompt: boolean = false;

  // Default prompt for v4.2 - synced with backend (database-agnostic)
  defaultSystemPrompt: string = `You are OptimaX, an autonomous AI agent for database analysis and SQL query generation.

!!!! CRITICAL RULE - READ THIS FIRST !!!!
NEVER run queries to "demonstrate capabilities" or "show examples".
If user asks "what can you do", just describe your capabilities - DO NOT RUN QUERIES.
Only run execute_sql when user asks for SPECIFIC data like "show me records" or "count entries".
Once you have the answer, STOP immediately - don't run additional queries.
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

{SCHEMA_SECTION}

WHEN TO USE TOOLS:
1. get_schema: ONLY if user asks about available tables/schema
2. execute_sql: ONLY when user explicitly requests specific data
3. After get_schema for a "what can you do" question, list capabilities and STOP - no execute_sql
4. After execute_sql returns useful data, formulate answer and STOP - don't run more queries

EFFICIENCY RULES:
- Use simple queries - avoid complex JOINs when possible
- Prefer direct table queries over complex joins
- STOP after getting useful results - don't keep exploring
- If you get an error, check schema and try ONE more time, then STOP

RULES:
- ALWAYS use schema prefix if required (check schema section above)
- Always use LIMIT for safety
- For date/time: EXTRACT(MONTH FROM column_name)
- Prefer COUNT/AVG/SUM/GROUP BY over raw rows
- Order DESC for most, ASC for least

RESPONSE:
- Answer the question and STOP
- Be concise and accurate
- Don't run multiple queries unless absolutely necessary`;

  // Preset prompts (removed - using single default only)
  presetPrompts: any[] = [];

  // UI state
  isLoading: boolean = false;
  saveInProgress: boolean = false;
  applyInProgress: boolean = false;

  // Validation and stats
  promptStats = { characters: 0, words: 0, lines: 0 };

  // Success/Error messages
  message: string = '';
  messageType: 'success' | 'error' | 'info' = 'info';

  ngOnInit(): void {
    this.loadSystemPrompt();
    this.loadActivePrompt();
  }

  loadSystemPrompt(): void {
    // Load from localStorage or use default
    const savedPrompt = localStorage.getItem('optimax-system-prompt');
    const savedSchemaOption = localStorage.getItem('optimax-use-dynamic-schema');

    this.systemPrompt = savedPrompt || this.defaultSystemPrompt;
    this.useDynamicSchema = savedSchemaOption !== null ? savedSchemaOption === 'true' : true;
    this.updateStats();
    this.showMessage('System prompt loaded', 'success');
  }

  loadActivePrompt(): void {
    this.chatService.getActiveSystemPrompt().subscribe({
      next: (response) => {
        this.activePromptInfo = response;
      },
      error: (err) => {
        console.error('Failed to load active prompt:', err);
      }
    });
  }

  savePrompt(): void {
    if (!this.systemPrompt.trim()) {
      this.showMessage('Prompt cannot be empty', 'error');
      return;
    }

    this.saveInProgress = true;

    // Save to backend permanently
    this.chatService.saveSystemPrompt(this.systemPrompt, this.useDynamicSchema).subscribe({
      next: (response) => {
        this.saveInProgress = false;
        this.showMessage('✅ Saved! Click "Apply Changes" to activate.', 'success');

        // Also save to localStorage for reference
        localStorage.setItem('optimax-system-prompt', this.systemPrompt);
        localStorage.setItem('optimax-use-dynamic-schema', this.useDynamicSchema.toString());

        // Emit to parent component
        this.promptChanged.emit(this.systemPrompt);

        // Reload active prompt info
        this.loadActivePrompt();
      },
      error: (err) => {
        this.saveInProgress = false;
        this.showMessage('❌ Failed to save: ' + err.message, 'error');
      }
    });
  }

  applyPrompt(): void {
    if (confirm('Apply the saved system prompt?\n\nThis will reload the agent and clear all active sessions.')) {
      this.applyInProgress = true;

      this.chatService.applySystemPrompt().subscribe({
        next: (response) => {
          this.applyInProgress = false;
          this.showMessage('✅ ' + response.message, 'success');

          // Dispatch event for chat interface to update
          const event = new CustomEvent('systemPromptChanged', {
            detail: { prompt: this.systemPrompt },
            bubbles: true
          });
          document.dispatchEvent(event);

          // Reload active prompt info
          this.loadActivePrompt();
        },
        error: (err) => {
          this.applyInProgress = false;
          this.showMessage('❌ Failed to apply: ' + err.message, 'error');
        }
      });
    }
  }

  resetPrompt(): void {
    if (confirm('Reset to default dynamic system prompt?\n\nThis will remove your custom prompt and apply default immediately.')) {
      this.saveInProgress = true;

      this.chatService.resetSystemPrompt().subscribe({
        next: (response) => {
          this.systemPrompt = this.defaultSystemPrompt;
          this.useDynamicSchema = true;
          localStorage.removeItem('optimax-system-prompt');
          localStorage.removeItem('optimax-use-dynamic-schema');
          this.updateStats();

          // Auto-apply after reset
          this.chatService.applySystemPrompt().subscribe({
            next: (applyResponse) => {
              this.saveInProgress = false;
              this.showMessage('✅ Reset to default and applied successfully!', 'success');
              this.promptChanged.emit(this.systemPrompt);

              // Dispatch event for chat interface to update
              const event = new CustomEvent('systemPromptChanged', {
                detail: { prompt: this.systemPrompt },
                bubbles: true
              });
              document.dispatchEvent(event);

              // Reload active prompt info
              this.loadActivePrompt();
            },
            error: (err) => {
              this.saveInProgress = false;
              this.showMessage('❌ Reset succeeded but apply failed: ' + err.message, 'error');
            }
          });
        },
        error: (err) => {
          this.saveInProgress = false;
          this.showMessage('❌ Failed to reset: ' + err.message, 'error');
        }
      });
    }
  }

  viewActivePrompt(): void {
    this.loadActivePrompt();
    this.showActivePrompt = true;
  }

  closeActivePrompt(): void {
    this.showActivePrompt = false;
  }

  hasChanges(): boolean {
    const savedPrompt = localStorage.getItem('optimax-system-prompt');
    return this.systemPrompt !== (savedPrompt || this.defaultSystemPrompt);
  }

  updateStats(): void {
    this.promptStats = this.calculateStats(this.systemPrompt);
  }

  private calculateStats(text: string) {
    return {
      characters: text.length,
      words: text.trim().split(/\s+/).filter(word => word.length > 0).length,
      lines: text.split('\n').length
    };
  }

  onPromptChange(): void {
    this.updateStats();
  }

  exportPrompt(): void {
    const data = {
      system_prompt: this.systemPrompt,
      version: '4.2',
      architecture: '4-Gate Intent Routing + Single Groq LLM (llama-3.3-70b-versatile)',
      exported_at: new Date().toISOString()
    };

    const dataStr = JSON.stringify(data, null, 2);
    const dataBlob = new Blob([dataStr], {type: 'application/json'});
    const link = document.createElement('a');
    link.href = URL.createObjectURL(dataBlob);
    link.download = `optimax-v4.2-prompt-${new Date().toISOString().split('T')[0]}.json`;
    link.click();
    this.showMessage('Prompt exported successfully', 'success');
  }

  testPrompt(): void {
    const testMessage = 'Show me top 10 passengers with most award points';

    this.showMessage('Testing prompt with sample query...', 'info');

    // Save the current prompt first
    this.savePrompt();

    // Close the modal and trigger test in main chat
    setTimeout(() => {
      this.closeModal();
      this.triggerTestInMainChat(testMessage);
    }, 1000);
  }

  private triggerTestInMainChat(testMessage: string): void {
    const event = new CustomEvent('testPrompt', {
      detail: { message: testMessage },
      bubbles: true
    });
    document.dispatchEvent(event);
  }

  private closeModal(): void {
    const event = new CustomEvent('closeSystemPromptManager', { bubbles: true });
    document.dispatchEvent(event);
  }

  isDifferentFromDefault(): boolean {
    return this.systemPrompt !== this.defaultSystemPrompt;
  }

  private showMessage(text: string, type: 'success' | 'error' | 'info'): void {
    this.message = text;
    this.messageType = type;
    setTimeout(() => {
      this.message = '';
    }, 5000);
  }

  copyToClipboard(): void {
    navigator.clipboard.writeText(this.systemPrompt).then(() => {
      this.showMessage('Prompt copied to clipboard', 'success');
    });
  }
}