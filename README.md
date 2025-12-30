# Ronak - Ask Me Anything

A single-page chat app that lets visitors ask questions about me (Ronak Vimal). The UI presents a profile panel on the left and a chat interface on the right. Under the hood, the app calls Groq's LLM twice per user message: first to classify intent, then to answer. Personal questions must use tool calling to fetch facts from `data/profile.json`, while non-personal questions are answered directly by the model. Messages render with Markdown and GitHub-flavored Markdown support.

## Tech Stack

- React 19 + TypeScript
- Vite 6
- Groq API (model: `openai/gpt-oss-120b`) via Cloudflare Pages Function
- React Markdown + remark-gfm for rich message rendering
- Lucide React for icons
- Tailwind via CDN + custom CSS in `index.html`

## How It Works

1. **Request in**: The browser POSTs `{ history, newMessage }` to `/api/chat`.
2. **Intent classifier**: The function sends a short summary of chat history + the new message to a classifier prompt. The model returns JSON `{ is_personal: boolean }` using a strict schema.
3. **Tool-enforced answering**:
   - If `is_personal` is true, the system prompt requires a tool call to `get_profile_info`.
   - If `is_personal` is false, the model answers normally.
4. **Tool calling**:
   - The model returns `tool_calls` with `key_path` or `key_paths`.
   - The function validates keys against the allowlist and reads values from `data/profile.json`.
   - If any key is invalid or missing, the response falls back to the missing-info string.
5. **Final answer**: The function appends tool results and asks the model for the final response.

## Intent Classifier

The classifier only decides whether a question is personal about Ronak Vimal. It uses:

- A dedicated system prompt and a strict JSON schema.
- A compact history summary (last `MAX_HISTORY_LENGTH` messages) to keep tokens down.
- The same model as the main responder by default (configurable via `CLASSIFIER_MODEL_NAME`).

## Tool Calling

The only allowed tool is `get_profile_info`. It is restricted to a fixed list of key paths (see `PROFILE_KEY_PATHS` in `functions/api/chat.ts`). The function rejects tool calls if:

- The tool name is not `get_profile_info`.
- Any requested key path is not in the allowlist.
- Any requested fact is missing or empty.

## API Key Rotation

The API route supports up to five Groq API keys. It starts with key #1 for each user session and rotates to the next key only on rate-limit errors. The chosen key index is stored in a session cookie so subsequent requests keep using the working key.

Required environment variables:

- `GROQ_API_KEY_1`
- `GROQ_API_KEY_2`
- `GROQ_API_KEY_3`
- `GROQ_API_KEY_4`
- `GROQ_API_KEY_5`

## Local Setup

**Prerequisites:** Node.js 18+ and npm

1. Clone the repo:
   `git clone <your-repo-url>`
2. Install dependencies:
   `npm install`
3. Create `.dev.vars` and add your Groq keys (for local Pages Functions):
   `GROQ_API_KEY_1=your_groq_api_key_here`
   `GROQ_API_KEY_2=your_groq_api_key_here`
   `GROQ_API_KEY_3=your_groq_api_key_here`
   `GROQ_API_KEY_4=your_groq_api_key_here`
   `GROQ_API_KEY_5=your_groq_api_key_here`
4. Update the profile data (used for personal Q&A):
   `data/profile.json`
5. (Optional) Replace `profile.jpeg` and update the contact line in `App.tsx`.
6. Start the Vite dev server:
   `npm run dev`
7. In another terminal, start the Pages Functions proxy:
   `npx wrangler pages dev --proxy 3000`
8. Open `http://localhost:8788` in your browser.

## Build & Preview

- `npm run build`
- `npm run preview`

## Deploy to Cloudflare Pages

1. Create a new Cloudflare Pages project and connect this repo.
2. Build command: `npm run build`
3. Build output directory: `dist`
4. Add `GROQ_API_KEY_1` through `GROQ_API_KEY_5` in Project Settings -> Environment Variables (Production + Preview).
5. Deploy. The API keys stay server-side in the Pages Function environment.

## Notes

- The browser only calls `/api/chat`; Groq API keys stay server-side in Cloudflare Pages environment variables.
