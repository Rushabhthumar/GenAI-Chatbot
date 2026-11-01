// app.ts - the chat screen. All state lives in signals; app.html re-renders automatically when they change.
import { Component, inject, signal } from '@angular/core'; // Angular building blocks.
import { takeUntilDestroyed } from '@angular/core/rxjs-interop'; // Auto-unsubscribe when the component is destroyed.
import { ChatSocketService } from './chat-socket.service'; // The WebSocket service.
import { ChatMessage, ChatMode, ServerEvent, StepEvent } from './chat.models'; // Types.

@Component({
  selector: 'app-root', // <app-root> in index.html.
  templateUrl: './app.html', // Template.
  styleUrl: './app.css', // Styles.
})
export class App {
  private readonly socket = inject(ChatSocketService); // Dependency injection.

  protected readonly connected = this.socket.connected; // Re-expose the connection signal to the template.
  protected readonly messages = signal<ChatMessage[]>([]); // All bubbles, oldest first.
  protected readonly question = signal(''); // Text in the input box.
  protected readonly mode = signal<ChatMode>('pipeline'); // Pipeline or agent.
  protected readonly busy = signal(false); // true while an answer is streaming.

  constructor() {
    this.socket.events$ // Listen to EVERYTHING the server pushes...
      .pipe(takeUntilDestroyed()) // ...until this component is destroyed...
      .subscribe((event) => this.handleEvent(event)); // ...and handle each event.
  }

  protected send(): void { // Enter key or Send button.
    const question = this.question().trim(); // Read the signal (call it like a function).
    if (!question || this.busy()) return; // Ignore empty input or double clicks.
    this.messages.update((list) => [ // Add two bubbles:
      ...list, // the old ones,
      { role: 'user', text: question, sources: [], steps: [], blocked: false }, // the question,
      { role: 'assistant', text: '', sources: [], steps: [], blocked: false }, // and an EMPTY answer that fills up live.
    ]);
    this.question.set(''); // Clear the input.
    this.busy.set(true); // Lock sending until "done".
    this.socket.send({ type: 'ask', question, mode: this.mode() }); // Only the question - the SERVER keeps the history.
  }

  protected newChat(): void { // "New chat" button.
    this.socket.send({ type: 'reset' }); // Server forgets the conversation.
    this.messages.set([]); // Clear the screen.
    this.busy.set(false); // Unlock.
  }

  private handleEvent(event: ServerEvent): void { // One server event -> update the last (assistant) bubble.
    switch (event.type) { // TypeScript narrows `event` to the right shape in each case.
      case 'step':
        this.updateLast((m) => ({ ...m, steps: this.upsertStep(m.steps, event) })); // Add/update the step in the trace.
        break;
      case 'sources':
        this.updateLast((m) => ({ ...m, sources: event.sources })); // Show the citation chips.
        break;
      case 'token':
        this.updateLast((m) => ({ ...m, text: m.text + event.text })); // Append the new piece of text.
        break;
      case 'blocked':
        this.updateLast((m) => ({ ...m, text: event.message, blocked: true })); // "query fails -> response".
        break;
      case 'error':
        this.updateLast((m) => ({ ...m, text: `⚠️ ${event.message}` })); // Show the error.
        break;
      case 'done':
        this.busy.set(false); // Answer finished - unlock sending.
        break;
    }
  }

  private upsertStep(steps: StepEvent[], step: StepEvent): StepEvent[] { // Replace a step with the same name, or add it.
    const exists = steps.some((s) => s.name === step.name); // Already in the list (e.g. 'running' -> now 'done')?
    return exists ? steps.map((s) => (s.name === step.name ? step : s)) : [...steps, step]; // New array = signals notice.
  }

  private updateLast(change: (m: ChatMessage) => ChatMessage): void { // Immutable update of the last bubble.
    this.messages.update((list) => {
      const last = list.at(-1); // The last bubble (or undefined).
      if (!last || last.role !== 'assistant') return list; // Safety: only ever change an assistant bubble.
      return [...list.slice(0, -1), change(last)]; // Copy without the last + the changed last one.
    });
  }
}
