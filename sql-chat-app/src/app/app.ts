import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { provideHttpClient, withFetch } from '@angular/common/http';
import { ChatInterfaceComponent } from './components/chat-interface/chat-interface';
import { LoadingScreenComponent } from './components/loading-screen/loading-screen';

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [CommonModule, ChatInterfaceComponent, LoadingScreenComponent],
  templateUrl: './app.html',
  styleUrl: './app.scss'
})
export class App implements OnInit {
  title = 'OptimaX SQL Assistant';
  isLoading = true;

  ngOnInit() {
    // Simulate loading time
    setTimeout(() => {
      this.isLoading = false;
    }, 3000); // 3 seconds loading
  }
}
