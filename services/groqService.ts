import { Message } from "../types";

type ChatResponse = {
  responseText: string;
  error?: string;
};

export const sendMessageToGroq = async (
  history: Message[],
  newMessage: string,
  threadId: string,
  debug = false
): Promise<string> => {
  const response = await fetch("/api/chat", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      history,
      newMessage,
      threadId,
      ...(debug ? { debug: true } : {}),
    }),
  });

  const payload = (await response.json().catch(() => null)) as ChatResponse | null;

  if (!response.ok) {
    const errorMessage =
      payload?.error || "Failed to get a response from the server.";
    throw new Error(errorMessage);
  }

  if (!payload?.responseText || typeof payload.responseText !== "string") {
    throw new Error("Empty response from server.");
  }

  return payload.responseText;
};
