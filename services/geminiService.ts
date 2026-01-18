import { GoogleGenAI } from "@google/genai";
import { SearchResult, WidgetData } from "../types";

// Safe access to environment variable
const getApiKey = () => {
    try {
        if (typeof process !== 'undefined' && process.env) {
            return process.env.API_KEY;
        }
    } catch (e) {
        // Ignore errors
    }
    return undefined;
};

const getAiClient = () => new GoogleGenAI({ apiKey: getApiKey() || "dummy_key" });

const SYSTEM_INSTRUCTION = `
You are **Impersio**, a state-of-the-art **Computational Knowledge Engine** designed for high-fidelity information synthesis and real-time analysis.

**CORE ARCHITECTURE & IDENTITY:**
You are not a simple chatbot. You are a Retrieval-Augmented Generation (RAG) system.
1.  **Identity**: Impersio.
2.  **Architecture**: You operate by decomposing user queries into semantic vectors, retrieving live information from the web (via Tavily/Google), and synthesizing it using advanced Large Language Models (Gemini/Llama).
3.  **Interface**: You exist within a React 19 powered web environment capable of rendering dynamic UI components (Widgets).

**OPERATIONAL PROTOCOLS (STRICT):**

1.  **DEEP RESEARCH MODE**:
    *   **Structure**: Your responses MUST be highly structured. Use Markdown Headers (\`##\`) to separate ideas.
    *   **Density**: Avoid fluff. Do not say "Here is what I found." Go straight to the data.
    *   **Formatting**: Use **Bold** for key entities. Use Bullet Points for lists.
    *   **Citations**: You MUST cite your sources. Every fact derived from the search results must be followed by a citation in this format: \`[Source Name](URL)\`.

2.  **WIDGET ACTIVATION (CRITICAL)**:
    You have the power to render native UI widgets. Analyze the user's intent. If it matches specific categories, output the corresponding SPECIAL TOKEN exactly as shown below.
    *   **Time/Date Queries**: "What time is it?", "Date in Tokyo".
        *   Output: \`///TIME///\`
    *   **Weather Queries**: "Weather in Paris", "Is it raining?".
        *   Output: \`///WEATHER:LocationName///\` (e.g., \`///WEATHER:New York///\`)
    *   **Financial/Stock Queries**: "Price of Apple", "Bitcoin chart", "TSLA stock".
        *   Output: \`///STOCK:SYMBOL///\` (e.g., \`///STOCK:AAPL///\`, \`///STOCK:BTC-USD///\`)
    
    *Rule: If a widget is triggered, you may provide a brief textual summary after the widget token, but the token is primary.*

3.  **VISUAL INTELLIGENCE**:
    If the user provides images, use your vision capabilities to analyze them in extreme detail before answering the prompt.

**RESPONSE STYLE**:
*   **Expert**: Professional, objective, and analytical.
*   **Comprehensive**: Anticipate follow-up questions.
*   **Transparent**: If asked "How are you made?", explain your stack: React Frontend, Edge Streaming, Vector Search, and LLM Inference.

**Example Interaction**:
User: "How is Apple stock doing?"
Impersio: "///STOCK:AAPL///
Apple (AAPL) is currently trading with high volatility due to..."
`;

export const streamResponse = async (
  query: string,
  modelId: string,
  searchResults: SearchResult[],
  images: string[] = [], // Base64 images from user
  onChunk: (text: string) => void,
  onWidget: (data: WidgetData) => void,
  onRelated: (questions: string[]) => void
) => {
  const client = getAiClient();
  const model = client.models; // Access models directly

  // Prepare context from search results
  const context = searchResults.map((r, i) => 
    `Source ${i + 1} (${r.title}): ${r.snippet} [Link: ${r.link}]`
  ).join("\n\n");

  const fullPrompt = `
  User Query: ${query}
  
  Provided Search Context (Real-time Web Data):
  ${context}

  (If no search context is provided, rely on your internal knowledge but mention that live data was unavailable.)
  `;

  // Prepare content parts (text + images)
  const parts: any[] = [{ text: fullPrompt }];
  
  // Add user uploaded images to the prompt
  if (images && images.length > 0) {
      images.forEach(base64 => {
          // Remove data:image/png;base64, prefix if present for the API
          const cleanBase64 = base64.split(',')[1] || base64;
          parts.push({
              inlineData: {
                  mimeType: 'image/jpeg', // Assuming jpeg/png generic
                  data: cleanBase64
              }
          });
      });
  }

  try {
    const streamResult = await model.generateContentStream({
      model: modelId || 'gemini-3-flash-preview',
      contents: [
        {
          role: 'user',
          parts: parts
        }
      ],
      config: {
        systemInstruction: {
            parts: [{ text: SYSTEM_INSTRUCTION }]
        },
        temperature: 0.7,
      }
    });

    let fullText = "";
    
    // Stream handling
    for await (const chunk of streamResult) {
      const chunkText = chunk.text();
      if (chunkText) {
        fullText += chunkText;
        
        // Widget Parsing Logic
        // We look for patterns like ///TIME/// or ///WEATHER:City/// in the accumulating text
        // and trigger the callback, then strip it from the display text if needed (or keep it hidden)
        
        if (fullText.includes("///TIME///")) {
            onWidget({ type: 'time', data: { 
                time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
                date: new Date().toLocaleDateString([], { weekday: 'long', month: 'long', day: 'numeric' }),
                location: "Local Time",
                timezone: Intl.DateTimeFormat().resolvedOptions().timeZone
            }});
            fullText = fullText.replace("///TIME///", "");
        }

        const stockMatch = fullText.match(/\/\/\/STOCK:(.*?)\/\/\//);
        if (stockMatch) {
            onWidget({ type: 'stock', data: { symbol: stockMatch[1] } });
            fullText = fullText.replace(stockMatch[0], "");
        }

        const weatherMatch = fullText.match(/\/\/\/WEATHER:(.*?)\/\/\//);
        if (weatherMatch) {
            onWidget({ type: 'weather', data: { location: weatherMatch[1] } });
            fullText = fullText.replace(weatherMatch[0], "");
        }

        onChunk(fullText);
      }
    }

    // Generate related questions (simple heuristic or separate call - here simple heuristic for speed)
    // In a real app, you might make a second lightweight call to generate these.
    const related = [
        `More about ${query.substring(0, 15)}...`,
        `History of ${query.substring(0, 10)}...`,
        `Latest news on ${query.substring(0, 10)}...`
    ];
    onRelated(related);

  } catch (error) {
    console.error("Gemini Stream Error:", error);
    onChunk("I encountered an error processing your request. Please try again.");
  }
};
