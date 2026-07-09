import { defineCollection, z } from 'astro:content';
import { glob } from 'astro/loaders';

const posts = defineCollection({
  loader: glob({ pattern: '**/index.md', base: './src/content/posts' }),
  schema: z.object({
    title: z.string(),
    description: z.string().optional().default(''),
    slug: z.string().optional(),
    date: z.coerce.date(),
    author: z.string().optional().default('Editorial'),
    keyword: z.string().optional(),
    thumbnail: z.string().optional(),
    tags: z.array(z.string()).optional().default([]),
  }),
});

export const collections = { posts };
