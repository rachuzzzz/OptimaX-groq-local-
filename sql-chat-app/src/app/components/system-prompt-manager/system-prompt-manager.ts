import { Component, OnInit, Output, EventEmitter } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';

@Component({
  selector: 'app-system-prompt-manager',
  standalone: true,
  imports: [CommonModule, FormsModule],
  template: `
    <div class="system-prompt-manager">
      <div class="manager-header">
        <h2>System Prompt Editor</h2>
        <p>Configure OptimaX v4.0 - Single Groq LLM (llama-3.3-70b-versatile)</p>
        <div class="architecture-info">
          <span class="badge">✨ Unified Architecture</span>
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
          Controls how Groq's llama-3.3-70b handles ALL tasks: greetings, SQL generation, data analysis, and chart recommendations.
        </p>

        <div class="warning-box" *ngIf="isDifferentFromDefault()">
          ⚠️ Custom prompt active - Using modified system prompt
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
              <li>Keep greeting detection rules clear (fast-path optimization)</li>
              <li>Define database columns and table schema</li>
              <li>Include few-shot examples for SQL generation</li>
              <li>Specify response format and tone</li>
              <li>Add safety rules (no DROP, DELETE, UPDATE)</li>
            </ul>
          </details>
        </div>

        <div class="actions">
          <button class="btn primary" (click)="savePrompt()" [disabled]="saveInProgress || !hasChanges()">
            {{ saveInProgress ? 'Saving...' : 'Apply Custom Prompt' }}
          </button>
          <button class="btn secondary" (click)="testPrompt()">Test Prompt</button>
          <button class="btn secondary" (click)="resetPrompt()" [disabled]="!isDifferentFromDefault()">
            Reset to Default
          </button>
          <button class="btn secondary" (click)="exportPrompt()">Export</button>
        </div>
      </div>

      <div class="info-panel">
        <h4>ℹ️ About v4.0 Architecture</h4>
        <p>OptimaX v4.0 uses a <strong>single LLM</strong> (Groq llama-3.3-70b-versatile) for all operations:</p>
        <ul>
          <li><strong>Greeting detection</strong> - Fast-path bypass (&lt;50ms)</li>
          <li><strong>SQL generation</strong> - Direct query creation</li>
          <li><strong>Data analysis</strong> - Result interpretation</li>
          <li><strong>Chart recommendations</strong> - Automatic visualization</li>
        </ul>
        <p class="note">
          <strong>Note:</strong> Changes are session-based and reset on page refresh.
          Export your prompt to save it permanently.
        </p>
      </div>
    </div>
  `,
  styleUrl: './system-prompt-manager.scss'
})
export class SystemPromptManagerComponent implements OnInit {
  @Output() promptChanged = new EventEmitter<string>();

  // Current prompt
  systemPrompt: string = '';

  // Default prompt for v4.0
  defaultSystemPrompt: string = `YOU ARE **OPTIMAX**, AN EXPERT AI ASSISTANT FOR ANALYZING UNITED STATES TRAFFIC ACCIDENT DATA (2016–2023). YOU HAVE READ-ONLY ACCESS TO A POSTGRESQL DATABASE WITH 7.7 MILLION RECORDS IN THE TABLE \`us_accidents\`.

YOUR PURPOSE IS TO **GENERATE SQL QUERIES**, **INTERPRET RESULTS**, AND **EXPLAIN INSIGHTS** CLEARLY AND CONCISELY.

---

### CAPABILITIES
1. **GENERATE ONE SQL QUERY** per user question — read-only, aggregate (COUNT, AVG, SUM, GROUP BY).
2. **INTERPRET RESULTS** in conversational English with context and significance.
3. **RECOMMEND CHART TYPES** when the user asks to "visualize", "plot", or "show as chart".

---

### INTENT CLASSIFICATION (STRICT)
1. **GREETING / CASUAL INTENT (NO TOOL USE)**
   - Triggered by short or social phrases such as:
     - "hi", "hello", "hey", "yo", "thanks", "thank you", "who are you", "help", "what can you do"
   - → IMMEDIATELY RESPOND FRIENDLY AND INFORMATIVE.
   - → DO NOT CALL OR EXECUTE ANY TOOL OR SQL QUERY.
   - 🔒 STRICT RULE: If user input is fewer than 6 words and matches greeting/casual intent → **NEVER EXECUTE ANY TOOL.**

2. **DATA QUESTION**
   - Phrases include: "show me", "how many", "list", "top", "compare", "count", "average", "trend"
   - → FORMULATE **ONE** SQL query (read-only, aggregate only).
   - → EXECUTE ONCE, THEN SUMMARIZE INSIGHTS CLEARLY.

---

### DATABASE COLUMNS
- Geographic: \`state\`, \`city\`, \`county\`, \`latitude\`, \`longitude\`
- Severity: \`severity\` (1–4, where 4 = most severe)
- Weather: \`weather_condition\`, \`temperature_f\`, \`visibility_mi\`, \`precipitation_in\`, \`humidity\`, \`wind_speed_mph\`
- Time: \`start_time\`, \`end_time\`, \`year\`, \`month\`, \`day\`, \`hour\`, \`day_of_week\`, \`is_weekend\`
- Road: \`street\`, \`junction\`, \`traffic_signal\`, \`crossing\`, \`railway\`, \`stop\`

---

### RESPONSE STYLE
- FRIENDLY + INFORMATIVE tone.
- PROVIDE CONTEXT ("California = 1.74 M accidents ≈ 22% of total").
- SUMMARIZE key insight + optional next step.

### WHAT NOT TO DO
- **DO NOT** execute tools during greetings.
- **DO NOT** run multiple SQLs for one query.
- **DO NOT** output > 50 rows.
- **DO NOT** skip ordering (use DESC for top values).`;

  // UI state
  isLoading: boolean = false;
  saveInProgress: boolean = false;

  // Validation and stats
  promptStats = { characters: 0, words: 0, lines: 0 };

  // Success/Error messages
  message: string = '';
  messageType: 'success' | 'error' | 'info' = 'info';

  constructor() {}

  ngOnInit(): void {
    this.loadSystemPrompt();
  }

  loadSystemPrompt(): void {
    // Load from localStorage or use default
    const savedPrompt = localStorage.getItem('optimax-system-prompt');
    this.systemPrompt = savedPrompt || this.defaultSystemPrompt;
    this.updateStats();
    this.showMessage('System prompt loaded', 'success');
  }

  savePrompt(): void {
    if (!this.systemPrompt.trim()) {
      this.showMessage('Prompt cannot be empty', 'error');
      return;
    }

    // Save to localStorage (session-based in v4.0)
    localStorage.setItem('optimax-system-prompt', this.systemPrompt);
    this.showMessage('✅ Custom prompt saved! Refresh page or create new session to apply.', 'success');

    // Emit to parent component
    this.promptChanged.emit(this.systemPrompt);
  }

  resetPrompt(): void {
    if (confirm('Reset to default OptimaX v4.0 system prompt?\n\nThis will restore the original prompt designed for optimal performance.')) {
      this.systemPrompt = this.defaultSystemPrompt;
      localStorage.removeItem('optimax-system-prompt');
      this.updateStats();
      this.showMessage('✅ Reset to default prompt. Refresh page or create new session to apply.', 'success');
      this.promptChanged.emit(this.systemPrompt);
    }
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
      version: '4.0',
      architecture: 'Single Groq LLM (llama-3.3-70b-versatile)',
      exported_at: new Date().toISOString()
    };

    const dataStr = JSON.stringify(data, null, 2);
    const dataBlob = new Blob([dataStr], {type: 'application/json'});
    const link = document.createElement('a');
    link.href = URL.createObjectURL(dataBlob);
    link.download = `optimax-v4-prompt-${new Date().toISOString().split('T')[0]}.json`;
    link.click();
    this.showMessage('Prompt exported successfully', 'success');
  }

  testPrompt(): void {
    const testMessage = 'Top 10 states by accident count';

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