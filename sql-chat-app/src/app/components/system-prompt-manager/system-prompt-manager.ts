import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { ChatService, SystemPromptsResponse } from '../../services/chat.service';

@Component({
  selector: 'app-system-prompt-manager',
  standalone: true,
  imports: [CommonModule, FormsModule],
  template: `
    <div class="system-prompt-manager">
      <div class="manager-header">
        <h2>System Prompts</h2>
        <p>Configure your AI models</p>
      </div>

      <div *ngIf="isLoading" class="loading">Loading...</div>

      <div *ngIf="message" class="message" [ngClass]="'message-' + messageType">
        {{ message }}
      </div>

      <div *ngIf="!isLoading" class="tabs">
        <button class="tab" [class.active]="activeTab === 'sql'" (click)="activeTab = 'sql'">
          SQL Generator
          <span class="modified" *ngIf="isDifferentFromDefault('sql')">●</span>
        </button>
        <button class="tab" [class.active]="activeTab === 'intent'" (click)="activeTab = 'intent'">
          Intent Classifier
          <span class="modified" *ngIf="isDifferentFromDefault('intent')">●</span>
        </button>
      </div>

      <div *ngIf="activeTab === 'sql'" class="editor">
        <h3>SQL Generation</h3>
        <p>Controls how CodeLlama generates PostgreSQL queries</p>
        <textarea class="prompt-input" [(ngModel)]="sqlPrompt" (ngModelChange)="onPromptChange()" placeholder="Loading current SQL prompt..."></textarea>
        <div class="actions">
          <button class="btn primary" (click)="savePrompt('sql')" [disabled]="saveInProgress">
            {{ saveInProgress ? 'Saving...' : 'Save Changes' }}
          </button>
          <button class="btn secondary" (click)="testPrompt('sql')">Test Prompt</button>
          <button class="btn secondary" (click)="resetPrompt('sql')">Reset to Default</button>
        </div>
      </div>

      <div *ngIf="activeTab === 'intent'" class="editor">
        <h3>Intent Classification</h3>
        <p>Controls how Phi-3 Mini classifies user messages</p>
        <textarea class="prompt-input" [(ngModel)]="intentPrompt" (ngModelChange)="onPromptChange()" placeholder="Loading current intent prompt..."></textarea>
        <div class="actions">
          <button class="btn primary" (click)="savePrompt('intent')" [disabled]="saveInProgress">
            {{ saveInProgress ? 'Saving...' : 'Save Changes' }}
          </button>
          <button class="btn secondary" (click)="testPrompt('intent')">Test Prompt</button>
          <button class="btn secondary" (click)="resetPrompt('intent')">Reset to Default</button>
        </div>
      </div>
    </div>
  `,
  styleUrl: './system-prompt-manager.scss'
})
export class SystemPromptManagerComponent implements OnInit {
  // Current prompts
  intentPrompt: string = '';
  sqlPrompt: string = '';

  // Default prompts for comparison
  defaultIntentPrompt: string = '';
  defaultSqlPrompt: string = '';

  // UI state
  activeTab: 'intent' | 'sql' = 'sql';
  isLoading: boolean = false;
  showDiff: boolean = false;
  saveInProgress: boolean = false;

  // Validation and stats
  promptStats = {
    intent: { characters: 0, words: 0, lines: 0 },
    sql: { characters: 0, words: 0, lines: 0 }
  };

  // Success/Error messages
  message: string = '';
  messageType: 'success' | 'error' | 'info' = 'info';

  constructor(private chatService: ChatService) {}

  ngOnInit(): void {
    this.loadSystemPrompts();
  }

  loadSystemPrompts(): void {
    this.isLoading = true;
    this.chatService.getSystemPrompts().subscribe({
      next: (response: SystemPromptsResponse) => {
        this.intentPrompt = response.intent_prompt;
        this.sqlPrompt = response.sql_prompt;
        this.defaultIntentPrompt = response.default_intent_prompt;
        this.defaultSqlPrompt = response.default_sql_prompt;
        this.updateStats();
        this.isLoading = false;
        this.showMessage('System prompts loaded successfully', 'success');
      },
      error: (error) => {
        console.error('Failed to load system prompts:', error);
        this.showMessage('Failed to load system prompts', 'error');
        this.isLoading = false;
      }
    });
  }

  savePrompt(modelType: 'intent' | 'sql'): void {
    const prompt = modelType === 'intent' ? this.intentPrompt : this.sqlPrompt;

    if (!prompt.trim()) {
      this.showMessage('Prompt cannot be empty', 'error');
      return;
    }

    this.saveInProgress = true;
    this.chatService.updateSystemPrompt({ model_type: modelType, prompt }).subscribe({
      next: (response) => {
        this.showMessage(`${modelType.charAt(0).toUpperCase() + modelType.slice(1)} prompt saved successfully`, 'success');
        this.saveInProgress = false;
      },
      error: (error) => {
        console.error('Failed to save prompt:', error);
        this.showMessage('Failed to save prompt', 'error');
        this.saveInProgress = false;
      }
    });
  }

  resetPrompt(modelType: 'intent' | 'sql'): void {
    const promptName = modelType === 'intent' ? 'Intent Classifier' : 'SQL Generator';

    if (confirm(`Are you sure you want to reset the ${promptName} prompt to default?\n\nThis will restore the original system prompt and cannot be undone.`)) {
      this.showMessage(`Resetting ${promptName} prompt to default...`, 'info');

      this.chatService.resetSystemPrompt(modelType).subscribe({
        next: (response) => {
          // Update the local prompt with the default value
          if (modelType === 'intent') {
            this.intentPrompt = this.defaultIntentPrompt;
          } else {
            this.sqlPrompt = this.defaultSqlPrompt;
          }
          this.updateStats();
          this.showMessage(`${promptName} prompt successfully reset to default`, 'success');
        },
        error: (error) => {
          console.error('Failed to reset prompt:', error);
          this.showMessage(`Failed to reset ${promptName} prompt. Please try again.`, 'error');
        }
      });
    }
  }

  resetAllPrompts(): void {
    if (confirm('Are you sure you want to reset ALL prompts to defaults?')) {
      this.chatService.resetAllSystemPrompts().subscribe({
        next: (response) => {
          this.intentPrompt = this.defaultIntentPrompt;
          this.sqlPrompt = this.defaultSqlPrompt;
          this.updateStats();
          this.showMessage('All prompts reset to defaults', 'success');
        },
        error: (error) => {
          console.error('Failed to reset prompts:', error);
          this.showMessage('Failed to reset prompts', 'error');
        }
      });
    }
  }

  validatePrompt(modelType: 'intent' | 'sql'): string[] {
    const prompt = modelType === 'intent' ? this.intentPrompt : this.sqlPrompt;
    const warnings: string[] = [];

    if (prompt.length < 50) {
      warnings.push('Prompt might be too short');
    }

    if (prompt.length > 3000) {
      warnings.push('Prompt might be too long');
    }

    if (modelType === 'sql') {
      if (!prompt.includes('{schema_text}')) {
        warnings.push('SQL prompt should include {schema_text} placeholder');
      }
      if (!prompt.includes('{question}')) {
        warnings.push('SQL prompt should include {question} placeholder');
      }
    }

    if (modelType === 'intent') {
      if (!prompt.toLowerCase().includes('sql_intent') || !prompt.toLowerCase().includes('chat_intent')) {
        warnings.push('Intent prompt should mention SQL_INTENT and CHAT_INTENT');
      }
    }

    return warnings;
  }

  updateStats(): void {
    this.promptStats.intent = this.calculateStats(this.intentPrompt);
    this.promptStats.sql = this.calculateStats(this.sqlPrompt);
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

  exportPrompts(): void {
    const data = {
      intent_prompt: this.intentPrompt,
      sql_prompt: this.sqlPrompt,
      exported_at: new Date().toISOString()
    };

    const dataStr = JSON.stringify(data, null, 2);
    const dataBlob = new Blob([dataStr], {type: 'application/json'});
    const link = document.createElement('a');
    link.href = URL.createObjectURL(dataBlob);
    link.download = `optimax-prompts-${new Date().toISOString().split('T')[0]}.json`;
    link.click();
  }

  importPrompts(event: any): void {
    const file = event.target.files[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (e) => {
        try {
          const data = JSON.parse(e.target?.result as string);
          if (data.intent_prompt) this.intentPrompt = data.intent_prompt;
          if (data.sql_prompt) this.sqlPrompt = data.sql_prompt;
          this.updateStats();
          this.showMessage('Prompts imported successfully', 'success');
        } catch (error) {
          this.showMessage('Failed to import prompts: Invalid file format', 'error');
        }
      };
      reader.readAsText(file);
    }
  }

  testPrompt(modelType: 'intent' | 'sql'): void {
    const testMessages = {
      intent: 'Test message: Show me accidents in Texas',
      sql: 'Test query: What are the top 5 states with most accidents?'
    };

    this.showMessage(`Testing ${modelType} prompt...`, 'info');

    // Save the current prompt first, then trigger a test
    this.savePrompt(modelType);

    // Close the modal and set the test message in the main chat
    setTimeout(() => {
      this.closeModal();
      // Emit an event or use a service to communicate with the parent component
      this.triggerTestInMainChat(testMessages[modelType]);
    }, 1000);
  }

  private triggerTestInMainChat(testMessage: string): void {
    // We'll emit this to the parent component
    const event = new CustomEvent('testPrompt', {
      detail: { message: testMessage },
      bubbles: true
    });
    document.dispatchEvent(event);
  }

  private closeModal(): void {
    // Close the system prompt manager
    const event = new CustomEvent('closeSystemPromptManager', { bubbles: true });
    document.dispatchEvent(event);
  }

  isDifferentFromDefault(modelType: 'intent' | 'sql'): boolean {
    if (modelType === 'intent') {
      return this.intentPrompt !== this.defaultIntentPrompt;
    } else {
      return this.sqlPrompt !== this.defaultSqlPrompt;
    }
  }

  private showMessage(text: string, type: 'success' | 'error' | 'info'): void {
    this.message = text;
    this.messageType = type;
    setTimeout(() => {
      this.message = '';
    }, 5000);
  }

  copyToClipboard(text: string): void {
    navigator.clipboard.writeText(text).then(() => {
      this.showMessage('Copied to clipboard', 'success');
    });
  }

  formatPrompt(modelType: 'intent' | 'sql'): void {
    if (modelType === 'intent') {
      this.intentPrompt = this.intentPrompt.trim();
    } else {
      this.sqlPrompt = this.sqlPrompt.trim();
    }
    this.updateStats();
  }
}