// chat-socket.service.ts - owns the ONE WebSocket connection to the Python backend (backend/main.py).
// Uses RxJS webSocket(): it opens the socket on first subscribe, parses JSON for us, and queues messages until connected.
import { Injectable, signal } from '@angular/core'; // Injectable = can be injected; signal = reactive value.
import { Observable, map, retry } from 'rxjs'; // RxJS operators.
import { webSocket, WebSocketSubject } from 'rxjs/webSocket'; // RxJS's WebSocket wrapper.
import { ClientMessage, ServerEvent } from './chat.models'; // Message types.

const WS_URL = 'ws://localhost:8000/ws/chat'; // Local backend. On Azure: 'wss://<your-app>.azurewebsites.net/ws/chat' (wss = TLS).

@Injectable({ providedIn: 'root' }) // One shared instance = one shared connection for the whole app.
export class ChatSocketService {
  readonly connected = signal(false); // true while the socket is open - the UI shows a green/red dot.

  private readonly socket: WebSocketSubject<ServerEvent | ClientMessage> = webSocket({ // A Subject: we can both send (next) and receive (subscribe).
    url: WS_URL, // Where to connect.
    openObserver: { next: () => this.connected.set(true) }, // Called when the connection opens.
    closeObserver: { next: () => this.connected.set(false) }, // Called when it closes (server down, network lost...).
  });

  readonly events$: Observable<ServerEvent> = this.socket.pipe( // The stream of server events for components to subscribe to.
    map((event) => event as ServerEvent), // Everything we RECEIVE is a ServerEvent.
    retry({ delay: 3000 }), // Connection dropped? Reconnect after 3 s (the server starts a fresh, empty history).
  );

  send(message: ClientMessage): void { // Send one message to the server.
    this.socket.next(message); // Serialised to JSON automatically. Queued if the socket is still connecting.
  }
}
