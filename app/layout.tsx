import type { Metadata } from 'next'
import { Plus_Jakarta_Sans } from 'next/font/google'
import './globals.css'

const plusJakarta = Plus_Jakarta_Sans({ subsets: ['latin'], weight: ['300','400','500','600','700','800'] })

export const metadata: Metadata = {
  title: 'Gorzen Ingestion - Universal Document Vector Pipeline',
  description: 'Transform any document collection into a searchable vector database with Pinecone in minutes',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <body className={plusJakarta.className}>
        <div className="min-h-screen bg-aurora">
          {children}
        </div>
      </body>
    </html>
  )
}