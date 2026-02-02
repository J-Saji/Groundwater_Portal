import { useState, useRef, useEffect } from 'react';
import axios from 'axios';
import { ChatMessage, MapContext } from '../types';

interface ChatBotProps {
  isOpen: boolean;
  onClose: () => void;
  mapContext: MapContext;
  backendURL: string;
  selectedAdvancedModule: string;
}

export default function ChatBot({
  isOpen,
  onClose,
  mapContext,
  backendURL,
  selectedAdvancedModule
}: ChatBotProps) {
  const [messages, setMessages] = useState<ChatMessage[]>([
    {
      role: "assistant",
      content: "Hello! I'm your AI-powered Groundwater Expert Assistant.\n\nI can help you with:\n• Live database queries (wells, GRACE, rainfall, aquifers)\n• Technical definitions and methodologies\n• Timeseries trend analysis\n• Interpretation of advanced modules\n\nFeel free to ask me anything about the data you're viewing!",
      timestamp: new Date()
    }
  ]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const handleSend = async () => {
    if (!input.trim() || isLoading) return;

    const userMessage: ChatMessage = {
      role: "user",
      content: input,
      timestamp: new Date()
    };

    const currentInput = input;
    setInput("");
    setMessages(prev => [...prev, userMessage]);
    setIsLoading(true);

    try {
      // Call the new RAG agent endpoint
      const response = await axios.post(`${backendURL}/api/agent`, {
        question: currentInput,
        map_context: mapContext
      });

      const assistantMessage: ChatMessage = {
        role: "assistant",
        content: response.data.answer,
        sourcesUsed: response.data.success ? 1 : 0, // Indicate if agent was successful
        timestamp: new Date()
      };

      setMessages(prev => [...prev, assistantMessage]);

      // If there's an error from the agent, show it
      if (!response.data.success && response.data.error) {
        console.warn("Agent returned error:", response.data.error);
      }
    } catch (error) {
      console.error("Chat Error:", error);

      const errorMessage: ChatMessage = {
        role: "assistant",
        content: "I apologize, but I encountered an error processing your request. Please ensure Ollama is running and try again.",
        timestamp: new Date()
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const getSuggestedQuestions = (): string[] => {
    const baseQuestions = [
      "What is GRACE satellite data?",
      "Explain groundwater depletion in India"
    ];

    const moduleQuestions: Record<string, string[]> = {
      'ASI': [
        "Which aquifer zones have the best storage potential?",
        "What does the ASI score mean for recharge suitability?",
        "How reliable is this ASI assessment?"
      ],
      'NETWORK_DENSITY': [
        "Where are the monitoring coverage gaps?",
        "Which areas have the strongest GWL signals?",
        "What does local density tell us about data quality?"
      ],
      'SASS': [
        "Which sites are most stressed right now?",
        "How does GRACE data relate to ground measurements here?",
        "What is causing the high stress scores?"
      ],
      'GRACE_DIVERGENCE': [
        "Why is GRACE diverging from ground measurements?",
        "What does positive/negative divergence indicate?",
        "Should we trust GRACE or wells more in this region?"
      ],
      'FORECAST': [
        "What does this forecast indicate for water security?",
        "How much is the GRACE contribution to the prediction?",
        "Which areas should prioritize intervention?"
      ],
      'RECHARGE': [
        "What structures are best for this region?",
        "How was the recharge potential calculated?",
        "Why are different structures recommended for different sites?"
      ],
      'SIGNIFICANT_TRENDS': [
        "Which sites have the most reliable declining trends?",
        "What does the p-value tell us about significance?",
        "Are these trends accelerating or stable?"
      ],
      'CHANGEPOINTS': [
        "What caused these structural breaks in GWL?",
        "Do changepoints align with policy or climate events?",
        "How should we interpret regime shifts?"
      ],
      'LAG_CORRELATION': [
        "What does the rainfall-GWL lag reveal about aquifer type?",
        "Why do some sites respond faster than others?",
        "How can lag information guide irrigation timing?"
      ],
      'HOTSPOTS': [
        "Where are the priority intervention zones?",
        "What connects sites in the same cluster?",
        "How should we address hotspot clusters?"
      ]
    };

    // Return module-specific questions if active
    if (selectedAdvancedModule && moduleQuestions[selectedAdvancedModule]) {
      return [...moduleQuestions[selectedAdvancedModule], "Explain this analysis in simple terms"];
    }

    // Default questions when viewing maps
    if (mapContext.active_layers.length > 0) {
      return [
        ...baseQuestions,
        "What patterns do you see in this map?",
        "Analyze the current data displayed"
      ];
    }

    return baseQuestions;
  };

  if (!isOpen) return null;

  return (
    <div className="fixed bottom-6 right-6 w-[420px] h-[650px] bg-white rounded-xl shadow-2xl border border-gray-200 flex flex-col z-[10000] overflow-hidden">
      {/* Header */}
      <div className="bg-gradient-to-r from-slate-700 via-slate-800 to-slate-900 text-white px-5 py-4 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 bg-blue-500 rounded-full flex items-center justify-center font-bold text-lg">
            AI
          </div>
          <div>
            <h3 className="font-semibold text-base">Expert Assistant</h3>
            <p className="text-xs text-slate-300">Powered by Gemma 3</p>
          </div>
        </div>
        <button
          onClick={onClose}
          className="text-slate-300 hover:text-white hover:bg-white/10 rounded-lg p-2 transition-all"
          aria-label="Close chat"
        >
          <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
          </svg>
        </button>
      </div>

      {/* Messages Container */}
      <div className="flex-1 overflow-y-auto p-5 space-y-4 bg-slate-50">
        {messages.map((msg, idx) => (
          <div key={idx} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
            <div className={`max-w-[85%] ${msg.role === 'user' ? '' : 'flex gap-2'}`}>
              {msg.role === 'assistant' && (
                <div className="w-8 h-8 bg-blue-500 rounded-full flex items-center justify-center text-white font-bold text-sm flex-shrink-0 mt-1">
                  AI
                </div>
              )}
              <div className={`rounded-2xl px-4 py-3 ${msg.role === 'user'
                ? 'bg-blue-600 text-white shadow-md'
                : 'bg-white text-slate-800 border border-slate-200 shadow-sm'
                }`}>
                <div className="text-[13px] leading-relaxed whitespace-pre-wrap">{msg.content}</div>
                {msg.sourcesUsed !== undefined && msg.sourcesUsed > 0 && (
                  <div className={`text-xs mt-2 pt-2 border-t ${msg.role === 'user' ? 'border-blue-500' : 'border-slate-200'} flex items-center gap-1`}>
                    <svg className="w-3 h-3" fill="currentColor" viewBox="0 0 20 20">
                      <path d="M9 4.804A7.968 7.968 0 005.5 4c-1.255 0-2.443.29-3.5.804v10A7.969 7.969 0 015.5 14c1.669 0 3.218.51 4.5 1.385A7.962 7.962 0 0114.5 14c1.255 0 2.443.29 3.5.804v-10A7.968 7.968 0 0014.5 4c-1.255 0-2.443.29-3.5.804V12a1 1 0 11-2 0V4.804z" />
                    </svg>
                    <span className={msg.role === 'user' ? 'text-blue-100' : 'text-slate-600'}>
                      Used {msg.sourcesUsed} knowledge source{msg.sourcesUsed !== 1 ? 's' : ''}
                    </span>
                  </div>
                )}
                <div className={`text-[10px] mt-2 ${msg.role === 'user' ? 'text-blue-200' : 'text-slate-400'}`}>
                  {msg.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                </div>
              </div>
            </div>
          </div>
        ))}

        {/* Loading Animation */}
        {isLoading && (
          <div className="flex justify-start">
            <div className="flex gap-2">
              <div className="w-8 h-8 bg-blue-500 rounded-full flex items-center justify-center text-white font-bold text-sm flex-shrink-0">
                AI
              </div>
              <div className="bg-white rounded-2xl px-4 py-3 border border-slate-200 shadow-sm">
                <div className="flex gap-1.5">
                  <div className="w-2 h-2 bg-slate-400 rounded-full animate-bounce"></div>
                  <div className="w-2 h-2 bg-slate-400 rounded-full animate-bounce" style={{ animationDelay: '0.15s' }}></div>
                  <div className="w-2 h-2 bg-slate-400 rounded-full animate-bounce" style={{ animationDelay: '0.3s' }}></div>
                </div>
              </div>
            </div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      {/* Suggested Questions (only show on first message) */}
      {messages.length === 1 && (
        <div className="px-5 py-3 border-t border-slate-200 bg-white">
          <div className="flex items-center gap-2 text-xs font-semibold text-slate-600 mb-2">
            <svg className="w-4 h-4 text-blue-500" fill="currentColor" viewBox="0 0 20 20">
              <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-8-3a1 1 0 00-.867.5 1 1 0 11-1.731-1A3 3 0 0113 8a3.001 3.001 0 01-2 2.83V11a1 1 0 11-2 0v-1a1 1 0 011-1 1 1 0 100-2zm0 8a1 1 0 100-2 1 1 0 000 2z" clipRule="evenodd" />
            </svg>
            <span>{selectedAdvancedModule ? `Suggested questions for ${selectedAdvancedModule}` : 'Suggested questions'}</span>
          </div>
          <div className="space-y-1.5 max-h-32 overflow-y-auto">
            {getSuggestedQuestions().map((q, idx) => (
              <button
                key={idx}
                onClick={() => setInput(q)}
                className="w-full text-left text-xs px-3 py-2 bg-slate-50 hover:bg-blue-50 rounded-lg transition-all text-slate-700 hover:text-blue-700 border border-slate-200 hover:border-blue-300"
              >
                {q}
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Active Layers Indicator */}
      {mapContext.active_layers.length > 0 && messages.length > 1 && (
        <div className="px-5 py-2.5 border-t border-slate-200 bg-slate-50">
          <div className="text-xs text-slate-600 flex items-start gap-2">
            <svg className="w-3.5 h-3.5 text-blue-500 mt-0.5 flex-shrink-0" fill="currentColor" viewBox="0 0 20 20">
              <path fillRule="evenodd" d="M5.05 4.05a7 7 0 119.9 9.9L10 18.9l-4.95-4.95a7 7 0 010-9.9zM10 11a2 2 0 100-4 2 2 0 000 4z" clipRule="evenodd" />
            </svg>
            <div>
              <span className="font-semibold">Active Context: </span>
              {mapContext.active_layers.join(', ')}
              {mapContext.region.state && ` • ${mapContext.region.state}`}
              {mapContext.region.district && ` → ${mapContext.region.district}`}
            </div>
          </div>
        </div>
      )}

      {/* Input Box */}
      <div className="p-4 border-t border-slate-200 bg-white">
        <div className="flex gap-2">
          <textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder="Ask about groundwater, GRACE data, or advanced modules..."
            className="flex-1 p-3 border border-slate-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 text-sm resize-none placeholder-slate-400 transition-all"
            rows={2}
            disabled={isLoading}
            maxLength={500}
          />
          <button
            onClick={handleSend}
            disabled={!input.trim() || isLoading}
            className="bg-blue-600 hover:bg-blue-700 disabled:bg-slate-300 disabled:cursor-not-allowed text-white px-5 rounded-lg font-medium transition-all flex items-center justify-center shadow-sm hover:shadow-md"
            aria-label="Send message"
          >
            {isLoading ? (
              <svg className="w-5 h-5 animate-spin" fill="none" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
              </svg>
            ) : (
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8" />
              </svg>
            )}
          </button>
        </div>

        {/* Character Count */}
        <div className="text-[10px] text-slate-400 mt-1.5 text-right">
          {input.length} / 500 characters
        </div>
      </div>
    </div>
  );
}