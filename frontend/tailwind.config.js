/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        charcoal: {
          950: '#0B0F0D', // deep background
          900: '#101412', // main background
          850: '#141A17',
          800: '#181F1B', // secondary background
          700: '#1D2521', // card background
          600: '#26322C', // borders
          500: '#34423A',
          400: '#4D5F54',
        },
        emerald: {
          DEFAULT: '#10B981', // primary emerald
          400: '#34D399',     // secondary emerald
          500: '#10B981',
          600: '#059669',
          900: '#064E3B',
          950: '#022C22',
        },
        txt: {
          main: '#F3F5F4',      // main text
          muted: '#9CA8A0',     // secondary text
          dim: '#6E7C73',       // subdued text
        },
        border: {
          subtle: '#26322C',
          active: '#34423A',
          emerald: 'rgba(16, 185, 129, 0.3)',
        }
      },
      fontFamily: {
        sans: ['Geist', 'Inter', '-apple-system', 'BlinkMacSystemFont', 'Segoe UI', 'Roboto', 'sans-serif'],
        mono: ['JetBrains Mono', 'Fira Code', 'Courier New', 'monospace'],
      },
      boxShadow: {
        'subtle-emerald': '0 0 25px -5px rgba(16, 185, 129, 0.15)',
        'card': '0 1px 3px 0 rgba(0, 0, 0, 0.4), 0 1px 2px -1px rgba(0, 0, 0, 0.4)',
      }
    },
  },
  plugins: [],
}

