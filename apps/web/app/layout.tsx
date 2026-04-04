import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Elephant ID — Human-in-the-Loop Elephant Identification",
  description:
    "Elephant ID is a platform for identifying individual African elephants from field photographs. It combines AI analysis, SEEK coding, and expert review to accelerate conservation workflows.",
  icons: {
    icon: "/favicon.svg",
  },
  openGraph: {
    title: "Elephant ID — Human-in-the-Loop Elephant Identification",
    description:
      "Combine machine learning, structured SEEK coding, and expert review to identify individual African elephants at scale.",
    type: "website",
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
