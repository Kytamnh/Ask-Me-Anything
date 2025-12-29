import React, { useState, useRef, useEffect } from 'react';
import { Message, Role } from './types';
import { sendMessageToGroq } from './services/groqService';
import { MessageBubble } from './components/MessageBubble';
import { ChatInput } from './components/ChatInput';

const App: React.FC = () => {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: 'welcome-msg',
      role: Role.MODEL,
      content: "Hi there! Ask me anything!",
      timestamp: Date.now()
    }
  ]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, isLoading]);

  const handleSendMessage = async (text: string) => {
    if (!text.trim()) return;

    setError(null);
    const newUserMessage: Message = {
      id: Date.now().toString(),
      role: Role.USER,
      content: text,
      timestamp: Date.now(),
    };

    setMessages((prev) => [...prev, newUserMessage]);
    setIsLoading(true);

    try {
      const responseText = await sendMessageToGroq(messages, text);

      const newBotMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: Role.MODEL,
        content: responseText,
        timestamp: Date.now(),
      };

      setMessages((prev) => [...prev, newBotMessage]);
    } catch (err: any) {
      setError(err.message || "Something went wrong.");
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="stage">
      {/* Left Branding Section */}
      <aside className="branding">
        <div className="flex flex-col items-center">
          <div className="profile-frame">
            {/* Placeholder image that looks like the user provided */}
             <img
               src="/profile.jpeg"
               alt="Ronak Vimal"
             />
          </div>
          <h1>Ronak Vimal</h1>
          <p className="branding-contact">+1 (516) 789-7228 | ronakvimal2003@gmail.com</p>
        </div>

        <div className="branding-footer">
          Powered by openai/gpt-oss-120b (Groq)
        </div>
      </aside>

      {/* Right Chat Section */}
      <section className="chat-container">
        <div className="message-stream">
          <div className="flex-1" /> {/* Push messages to bottom initially if few */}
          
          {messages.map((msg) => (
            <MessageBubble key={msg.id} message={msg} />
          ))}

          {/* Loading Indicator */}
          {isLoading && (
            <div className="bubble ai" style={{ width: 'fit-content' }}>
               <div className="flex items-center gap-2">
                 <span className="w-2 h-2 bg-[var(--ink)] rounded-full animate-bounce" style={{ opacity: 0.5 }}></span>
                 <span className="w-2 h-2 bg-[var(--ink)] rounded-full animate-bounce" style={{ animationDelay: '150ms', opacity: 0.5 }}></span>
                 <span className="w-2 h-2 bg-[var(--ink)] rounded-full animate-bounce" style={{ animationDelay: '300ms', opacity: 0.5 }}></span>
               </div>
            </div>
          )}

          {/* Error Message */}
          {error && (
            <div className="text-center py-2 text-red-500 bg-red-50 rounded-lg mx-10 text-sm">
              {error}
            </div>
          )}

          <div ref={messagesEndRef} />
        </div>

        <ChatInput onSend={handleSendMessage} isLoading={isLoading} />
      </section>
    </div>
  );
};

export default App;
