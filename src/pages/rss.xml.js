import rss from '@astrojs/rss';
import { getCollection } from 'astro:content';

export async function GET(context) {
  const posts = (await getCollection('posts'))
    .filter((p) => !p.data.draft)
    .sort((a, b) => b.data.date.valueOf() - a.data.date.valueOf());
  return rss({
    title: 'Little Developer — AI Tech Blog',
    description:
      'Measured deep learning: PyTorch performance, IMU sensor models, and deployment pitfalls, verified with real experiments on real hardware.',
    site: context.site,
    items: posts.map((p) => ({
      title: p.data.title,
      description: p.data.description,
      pubDate: p.data.date,
      link: `/posts/${p.data.slug ?? p.id}/`,
    })),
  });
}
