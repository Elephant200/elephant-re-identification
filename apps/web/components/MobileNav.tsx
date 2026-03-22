"use client";

import { useState } from "react";
import { Menu, X } from "lucide-react";

export function MobileNav() {
  const [navOpen, setNavOpen] = useState(false);

  return (
    <>
      <button
        className="md:hidden text-[#4a3728]"
        onClick={() => setNavOpen(!navOpen)}
        aria-label="Toggle navigation"
      >
        {navOpen ? <X size={20} /> : <Menu size={20} />}
      </button>

      {navOpen && (
        <div className="md:hidden fixed top-16 left-0 right-0 z-50 bg-[#faf7f2] border-b border-[#e8dfc8] px-6 py-4 flex flex-col gap-4">
          <a
            href="#problem"
            className="text-sm text-[#4a3728]"
            onClick={() => setNavOpen(false)}
          >
            The Problem
          </a>
          <a
            href="#workflow"
            className="text-sm text-[#4a3728]"
            onClick={() => setNavOpen(false)}
          >
            How It Works
          </a>
          <a
            href="#seek"
            className="text-sm text-[#4a3728]"
            onClick={() => setNavOpen(false)}
          >
            SEEK Coding
          </a>
          <a
            href="#features"
            className="text-sm text-[#4a3728]"
            onClick={() => setNavOpen(false)}
          >
            Features
          </a>
          <a
            href="#contact"
            className="text-sm text-[#2d5a27] font-medium"
            onClick={() => setNavOpen(false)}
          >
            Request Access
          </a>
        </div>
      )}
    </>
  );
}
