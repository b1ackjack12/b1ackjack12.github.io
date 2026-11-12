import { defineConfig } from 'astro/config';
import sitemap from '@astrojs/sitemap';
import tailwindcss from '@tailwindcss/vite';

// Internal links in markdown are written as "/posts/slug"; GitHub Pages serves
// "/posts/slug/index.html" and 301-redirects the slash-less form. Append the slash
// at build time so every internal link resolves directly (Search Console reported
// the redirects as "page with redirect").
function rehypeTrailingSlash() {
  const fix = (href) => {
    if (typeof href !== 'string' || !href.startsWith('/') || href.startsWith('//')) return href;
    const m = href.match(/^([^?#]*)(.*)$/);
    const path = m[1];
    const rest = m[2];
    if (path.endsWith('/') || /\.[a-z0-9]+$/i.test(path)) return href;
    return `${path}/${rest}`;
  };
  const walk = (node) => {
    if (node.type === 'element' && node.tagName === 'a' && node.properties && node.properties.href) {
      node.properties.href = fix(node.properties.href);
    }
    if (Array.isArray(node.children)) node.children.forEach(walk);
  };
  return (tree) => walk(tree);
}

export default defineConfig({
  site: 'https://b1ackjack12.github.io',
  base: '/',
  trailingSlash: 'always',
  integrations: [sitemap()],
  markdown: { rehypePlugins: [rehypeTrailingSlash] },
  vite: { plugins: [tailwindcss()] },
});
