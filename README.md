# Ronak - Ask Me Anything

A single-page chat app that lets visitors ask questions about Ronak Vimal. The UI presents a profile panel on the left and a chat interface on the right. Under the hood, the app calls Groq's LLM to classify each message as personal vs. general. Personal questions trigger a tool call that pulls facts from `data/profile.json`, while non-personal questions are answered directly by the model. Messages render with Markdown and GitHub-flavored markdown support.


## Tech Stack

- React 19 + TypeScript
- Vite 6
- Groq SDK (model: `openai/gpt-oss-120b`)
- React Markdown + remark-gfm for rich message rendering
- Lucide React for icons
- Tailwind via CDN + custom CSS in `index.html`

## Local Setup

**Prerequisites:** Node.js 18+ and npm

1. Clone the repo:
   `git clone <your-repo-url>`
2. Install dependencies:
   `npm install`
3. Create `.env.local` and add your Groq key:
   `GROQ_API_KEY=your_groq_api_key_here`
4. Update the profile data (used for personal Q&A):
   `data/profile.json`
5. (Optional) Replace `profile.jpeg` and update the contact line in `App.tsx`.
6. Start the dev server:
   `npm run dev`
7. Open `http://localhost:3000` in your browser.

## Build & Preview

- `npm run build`
- `npm run preview`

## Notes

- The app calls Groq directly from the browser, so keep your API key private and avoid deploying this without a server-side proxy if you need strict key security.
