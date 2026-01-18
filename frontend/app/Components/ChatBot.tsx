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
      content: "Hello! I'm your Groundwater and Remote Sensing expert. Ask me anything about GRACE data, rainfall patterns, groundwater levels, aquifer systems, or advanced modules!",
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
      // Prepare chat history (last 6 messages, excluding current)
      const historyPayload = messages.slice(-6).map(msg => ({
        role: msg.role,
        content: msg.content
      }));

      const response = await axios.post(`${backendURL}/api/chat`, {
        question: currentInput,
        map_context: mapContext,
        chat_history: historyPayload
      });

      const assistantMessage: ChatMessage = {
        role: "assistant",
        content: response.data.answer,
        sourcesUsed: response.data.sources_used,
        timestamp: new Date()
      };

      setMessages(prev => [...prev, assistantMessage]);
    } catch (error) {
      console.error("Chat Error:", error);
      
      const errorMessage: ChatMessage = {
        role: "assistant",
        content: "Sorry, I encountered an error processing your question. Please try again.",
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
    <div className="fixed bottom-6 right-6 w-96 h-[600px] bg-white rounded-2xl shadow-2xl border-2 border-blue-300 flex flex-col z-[10000]">
      {/* Header */}
      <div className="bg-gradient-to-r from-blue-600 to-green-600 text-white p-4 rounded-t-2xl flex items-center justify-between">
        <div>
          <h3 className="font-bold text-lg">AI Expert Assistant</h3>
          <p className="text-xs text-blue-100">Powered by Llama3.1</p>
        </div>
        <button
          onClick={onClose}
          className="text-white hover:bg-white/20 rounded-lg p-2 transition-all"
        >
          ✕
        </button>
      </div>

      {/* Messages Container */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {messages.map((msg, idx) => (
          <div key={idx} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
            <div className={`max-w-[80%] rounded-2xl p-3 ${
              msg.role === 'user'
                ? 'bg-blue-500 text-white'
                : 'bg-gray-100 text-gray-800 border border-gray-200'
            }`}>
              <div className="text-sm whitespace-pre-wrap">{msg.content}</div>
              {msg.sourcesUsed !== undefined && msg.sourcesUsed > 0 && (
                <div className="text-xs mt-2 opacity-70">
                  📚 Used {msg.sourcesUsed} knowledge source{msg.sourcesUsed !== 1 ? 's' : ''}
                </div>
              )}
              <div className="text-xs mt-1 opacity-60">
                {msg.timestamp.toLocaleTimeString()}
              </div>
            </div>
          </div>
        ))}
        
        {/* Loading Animation */}
        {isLoading && (
          <div className="flex justify-start">
            <div className="bg-gray-100 rounded-2xl p-3 border border-gray-200">
              <div className="flex gap-1">
                <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce"></div>
                <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0.1s' }}></div>
                <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0.2s' }}></div>
              </div>
            </div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      {/* Suggested Questions (only show on first message) */}
      {messages.length === 1 && (
        <div className="px-4 py-2 border-t border-gray-200 bg-gray-50">
          <div className="text-xs font-semibold text-gray-600 mb-2">
            {selectedAdvancedModule ? `💡 Ask about ${selectedAdvancedModule}:` : '💡 Try asking:'}
          </div>
          <div className="space-y-1 max-h-32 overflow-y-auto">
            {getSuggestedQuestions().map((q, idx) => (
              <button
                key={idx}
                onClick={() => setInput(q)}
                className="w-full text-left text-xs p-2 bg-blue-50 hover:bg-blue-100 rounded-lg transition-all text-blue-700 border border-blue-200"
              >
                {q}
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Active Layers Indicator */}
      {mapContext.active_layers.length > 0 && messages.length > 1 && (
        <div className="px-4 py-2 border-t border-gray-200 bg-blue-50">
          <div className="text-xs text-blue-700">
            <strong>Active Context:</strong> {mapContext.active_layers.join(', ')}
            {mapContext.region.state && ` | ${mapContext.region.state}`}
            {mapContext.region.district && ` → ${mapContext.region.district}`}
          </div>
        </div>
      )}

      {/* Input Box */}
      <div className="p-4 border-t border-gray-200 bg-white rounded-b-2xl">
        <div className="flex gap-2">
          <textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder="Ask about groundwater, GRACE, or advanced modules..."
            className="flex-1 p-3 border-2 border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 text-sm resize-none"
            rows={2}
            disabled={isLoading}
          />
          <button
            onClick={handleSend}
            disabled={!input.trim() || isLoading}
            className="bg-blue-500 hover:bg-blue-600 disabled:bg-gray-300 disabled:cursor-not-allowed text-white px-4 py-2 rounded-lg font-semibold transition-all flex items-center justify-center"
          >
            {isLoading ? '⏳' : '📤'}
          </button>
        </div>
        
        {/* Character Count */}
        <div className="text-xs text-gray-500 mt-1 text-right">
          {input.length} / 500
        </div>
      </div>
    </div>
  );
}