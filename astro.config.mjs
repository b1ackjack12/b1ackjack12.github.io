import { defineConfig } from 'astro/config';
import sitemap from '@astrojs/sitemap';

// GitHub Pages 기준 site/base. 커스텀 도메인 연결 시 수정.
export default defineConfig({
  site: 'https://b1ackjack12.github.io',
  base: '/',
  integrations: [sitemap()],
});
