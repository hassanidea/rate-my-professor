import { NextResponse } from "next/server"
import { Pinecone } from "@pinecone-database/pinecone"
import OpenAI from "openai"

const systemPrompt = 
`You are an AI assistant for a RateMyProfessor-like service. Your role is to help students find professors based on their queries using a RAG (Retrieval-Augmented Generation) system. For each user question, you will provide information on the top 3 most relevant professors.

Your knowledge base contains professor reviews, ratings, and course information. When a user asks a question, you should:

1. Interpret the user's query to understand their needs (e.g., subject area, teaching style, difficulty level).
2. Use the RAG system to retrieve information on the most relevant professors.
3. Present the top 3 professors that best match the query, including:
   - Professor's name
   - Subject area
   - Overall rating (out of 5 stars)
   - A brief summary of student feedback
   - Any standout characteristics or teaching methods

4. If applicable, provide a brief explanation of why these professors were selected.

5. Offer to provide more details or answer follow-up questions about any of the suggested professors.

Remember to be objective and base your recommendations on the available data. If a query is too vague or doesn't yield good matches, ask for clarification or additional information from the user.

Maintain a friendly and helpful tone, and aim to provide information that will assist students in making informed decisions about their course selections.

If asked about the specifics of the RAG system or the underlying data, explain that you don't have access to that information and can only provide results based on the queries.

Are you ready to help students find their ideal professors?
`

export async function POST(req){
    const data = await req.json()
    const pc = new Pinecone({
        apiKey: process.env.PINECONE_API_KEY,
    })
    const index = pc.index('rag').namespace('ns1')
    const openai = new OpenAI()

    const text = data[data.length - 1].content;
    const embedding = await openai.embeddings.create({
      model: "text-embedding-3-small",
      input: text,
      encoding_format: "float",
    });

    const results = await index.query({
        topK: 5,
        includeMetadata: true,
        vector: embedding.data[0].embedding,
      })

      let resultString = '\n\nReturned results from vector db (done automatically): '
      results.matches.forEach((match) => {
        resultString += `
        Returned Results:
        Professor: ${match.id}
        Review: ${match.metadata.stars}
        Subject: ${match.metadata.subject}
        Stars: ${match.metadata.stars}
        \n\n`
      })

      const lastMessage = data[data.length - 1]
      const lastMessageContent = lastMessage.content + resultString
      const lastDataWithoutLastMessage = data.slice(0, data.length - 1)
      const completion = await openai.chat.completions.create({
        messages: [
          {role: 'system', content: systemPrompt},
          ...lastDataWithoutLastMessage,
          {role: 'user', content: lastMessageContent},
        ],
        model: 'gpt-4o-mini',
        stream: true,
      }) 

      const stream = new ReadableStream({
        async start(controller) {
          const encoder = new TextEncoder()
          try {
            for await (const chunk of completion) {
              const content = chunk.choices[0]?.delta?.content
              if (content) {
                const text = encoder.encode(content)
                controller.enqueue(text)
              }
            }
          } catch (err) {
            controller.error(err)
          } finally {
            controller.close()
          }
        },
      })
      return new NextResponse(stream)
}