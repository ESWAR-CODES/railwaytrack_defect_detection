/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        navy: {
          DEFAULT: "#0B132B",
          dark: "#070D1E",
          surface: "#1C2541",
          light: "#243258",
        },
        gold: {
          DEFAULT: "#F5A623",
          hover: "#D98E16",
          light: "#FEF3C7",
        },
        ivory: {
          DEFAULT: "#FDFBF7",
          dark: "#F4EFE6",
        },
        surface: {
          DEFAULT: "#F3F4F6",
          muted: "#E5E7EB",
          card: "#FFFFFF",
        },
        telemetry: {
          low: "#D97706",
          medium: "#EA580C",
          high: "#DC2626",
          healthy: "#10B981",
        },
        text: {
          primary: "#0F172A",
          muted: "#64748B",
          inverse: "#F8FAFC",
        },
        border: {
          muted: "#E2E8F0",
          dark: "#1E293B",
        },
      },
      fontFamily: {
        serif: ["'Playfair Display'", "Georgia", "serif"],
        sans: ["'Inter'", "-apple-system", "BlinkMacSystemFont", "sans-serif"],
        mono: ["'JetBrains Mono'", "monospace"],
      },
      borderRadius: {
        btn: "4px",
        card: "8px",
      },
      boxShadow: {
        editorial: "0 4px 20px -2px rgba(11, 19, 43, 0.08)",
        card: "0 2px 12px rgba(11, 19, 43, 0.06)",
        glow: "0 0 15px rgba(245, 166, 35, 0.25)",
      },
    },
  },
  plugins: [],
};
