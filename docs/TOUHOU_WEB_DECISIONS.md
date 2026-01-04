# touhou-web Decision Log (Private)

## Project Overview
Website for touhou.amadeuswoo.com integrating:
- touhou-composition-analysis (379 tracks, 110+ features, UMAP visualization)
- touhou-style-classifier (89.5% accuracy Random Forest, ONNX export)

## Tech Stack Decision

| Choice | Alternative | Rationale |
|--------|-------------|-----------|
| Next.js 14 | Vite/React | SSG support, Vercel native, App Router |
| Tailwind CSS | CSS Modules | Fast styling, consistent with modern projects |
| TypeScript | JavaScript | Type safety, better DX |
| ONNX Runtime Web | Server-side inference | No backend needed, free tier friendly |
| Vercel | Netlify, GitHub Pages | Native Next.js support, easy subdomain |

## Pages

### 1. Landing (`/`)
- Hero section with project title
- Key findings summary (era evolution, embeddings comparison)
- Navigation to Explorer and Classifier

### 2. Explorer (`/explorer`)
- Interactive UMAP scatter plot (379 tracks)
- Color by: Era, Game, Stage Position
- Hover: Track name, game, features
- Data source: `umap_coords.json` (pre-computed)

### 3. Classifier (`/classifier`)
- Audio file upload (drag & drop)
- Client-side feature extraction (Meyda.js or Web Audio API)
- ONNX inference for circle classification
- Display: Predicted circle + confidence scores

## Data Files to Include

| File | Size | Purpose |
|------|------|---------|
| `umap_coords.json` | ~50KB | Pre-computed 2D coordinates for 379 tracks |
| `rf_classifier.onnx` | ~2MB | Trained Random Forest model |
| `label_map.json` | <1KB | Class index → circle name |
| `catalog.json` | ~100KB | Track metadata for hover info |

## Timeline

- [ ] Scaffold Next.js project
- [ ] Build landing page
- [ ] Build Explorer page with UMAP visualization
- [ ] Build Classifier page with ONNX inference
- [ ] Deploy to Vercel
- [ ] Configure touhou.amadeuswoo.com subdomain

## Open Questions (Need User Input)

1. **Color scheme**: Dark theme (matches Touhou aesthetic) or light?
2. **Additional pages**: Include game-by-game breakdowns from composition analysis?
3. **Diffusion showcase**: Include generated spectrograms (toy but interesting)?

---
*Last updated: 2026-01-01*
