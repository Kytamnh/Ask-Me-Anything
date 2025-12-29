import React, { useState, useRef, useEffect } from 'react';
import { SendHorizontal } from 'lucide-react';
import { PLACEHOLDER_TEXT } from '../constants';

interface ChatInputProps {
  onSend: (text: string) => void;
  isLoading: boolean;
}

export const ChatInput: React.FC<ChatInputProps> = ({ onSend, isLoading }) => {
  const [text, setText] = useState('');
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const handleSubmit = (e?: React.FormEvent) => {
    e?.preventDefault();
    if (!text.trim() || isLoading) return;
    
    onSend(text);
    setText('');
    
    // Reset height
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  // Auto-resize textarea
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      // Slight adjustment for the new padding
      const scrollHeight = textareaRef.current.scrollHeight;
      textareaRef.current.style.height = `${Math.min(scrollHeight, 100)}px`;
    }
  }, [text]);

  return (
    <div className="input-region">
      <form onSubmit={handleSubmit} className="input-wrapper">
        <textarea
          ref={textareaRef}
          value={text}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder={PLACEHOLDER_TEXT}
          disabled={isLoading}
          rows={1}
          className="input-field"
        />
        <button
          type="submit"
          disabled={!text.trim() || isLoading}
          className="send-btn"
          aria-label="Send"
        >
          <SendHorizontal size={22} className={text.trim() ? "translate-x-[1px]" : ""} />
        </button>
      </form>
    </div>
  );
};