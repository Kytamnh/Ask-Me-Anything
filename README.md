# Ronak - Ask Me Anything

A single-page chat app that lets visitors ask questions about me (Ronak Vimal). The UI presents a profile panel on the left and a chat interface on the right. Under the hood, the app calls Groq's LLM to classify each message as personal vs. general. Personal questions trigger a tool call that pulls facts from `data/profile.json`, while non-personal questions are answered directly by the model. Messages render with Markdown and GitHub-flavored markdown support.


## Tech Stack

- React 19 + TypeScript
- Vite 6
- Groq API (model: `openai/gpt-oss-120b`) via Cloudflare Pages Function
- React Markdown + remark-gfm for rich message rendering
- Lucide React for icons
- Tailwind via CDN + custom CSS in `index.html`

## Local Setup

**Prerequisites:** Node.js 18+ and npm

1. Clone the repo:
   `git clone <your-repo-url>`
2. Install dependencies:
   `npm install`
3. Create `.dev.vars` and add your Groq key (for local Pages Functions):
   `GROQ_API_KEY=your_groq_api_key_here`
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
4. Add `GROQ_API_KEY` in Project Settings -> Environment Variables (Production + Preview).
5. Deploy. The API key stays server-side in `functions/api/chat.ts`.

## Notes

- The browser only calls `/api/chat`; the Groq API key stays in Cloudflare Pages environment variables.
