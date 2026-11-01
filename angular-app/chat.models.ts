// chat.models.ts - TypeScript types for every WebSocket message. They mirror the Python side (backend/models.py + main.py).

export type ChatMode = 'pipeline' | 'agent'; // pipeline = fixed diagram flow, agent = the LLM chooses its tools.

export interface Source { // One citation chip under an answer.
  number: number; // [1], [2]... same number the LLM uses in the answer text.
  fileName: string; // Which document.
  page: number; // Which page.
  score: number | null; // Cosine similarity (null for SQL/metadata results).
}

export interface StepEvent { // One step of the diagram, shown live in the "trace".
  type: 'step';
  name: string; // e.g. 'guardrails', 'intent_routing', 'semantic_search', 'tool:metadata_search'.
  status: 'running' | 'done'; // Started or finished.
  detail: string; // e.g. 'hybrid | search: ... | filters: {...}'.
}

export type ServerEvent = // Everything the server can push. A "discriminated union": `type` tells which shape it is.
  | StepEvent
  | { type: 'sources'; sources: Source[] } // Citations (sent before the answer text).
  | { type: 'token'; text: string } // One piece of the answer.
  | { type: 'blocked'; category: string; message: string } // Guardrails stopped the question ("query fails").
  | { type: 'error'; message: string } // Something failed on the server.
  | { type: 'done' } // The answer is complete.
  | { type: 'reset_done' }; // The server forgot the conversation.

export type ClientMessage = // Everything the browser can send.
  | { type: 'ask'; question: string; mode: ChatMode } // A question.
  | { type: 'reset' }; // "New chat".

export interface ChatMessage { // One bubble on screen.
  role: 'user' | 'assistant'; // Who wrote it.
  text: string; // The text (assistant text grows while tokens arrive).
  sources: Source[]; // Citations (assistant only).
  steps: StepEvent[]; // Live trace of the diagram steps (assistant only).
  blocked: boolean; // true = guardrails blocked the question.
}
