/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        background: "#0d0f12",
        panel: "#161b22",
        primary: "#3b82f6",
        danger: "#ef4444",
        warning: "#f59e0b",
        success: "#10b981",
        border: "#2d3748"
      }
    },
  },
  plugins: [],
}
