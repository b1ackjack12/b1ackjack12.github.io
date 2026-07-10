import { defineCollection, z } from 'astro:content';
import { glob } from 'astro/loaders';

const posts = defineCollection({
  loader: glob({ pattern: '**/index.md', base: './src/content/posts' }),
  schema: ({ image }) => z.object({
    title: z.string(),
    description: z.string().optional().default(''),
    slug: z.string().optional(),
    date: z.coerce.date(),
    author: z.string().optional().default('Editorial'),
    keyword: z.string().optional(),
    thumbnail: image().optional(),
    tags: z.array(z.string()).optional().default([]),
    draft: z.boolean().optional().default(false),
  }),
});

export const collections = { posts };
