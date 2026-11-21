# SERPENTIS AI Website
React + Vite website. Pages: Home, Playground UI, API Keys, Account, Admin preview, Roadmap.

Run web UI: npm install && npm run dev

Supabase auth setup requires VITE_SUPABASE_URL and VITE_SUPABASE_ANON_KEY. Enable GitHub and Google providers and set redirect URLs in Supabase. VITE_ADMIN_EMAIL is only a client preview gate; real admin authorization must be enforced server-side. API-key issuance is intentionally not fake/client-generated. Playground is placeholder and does not call an AI service. Model integration will come later.