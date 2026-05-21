-- Per-lecture conversational chat history (active thread only).
create table if not exists public.chat_history (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users (id) on delete cascade,
  doc_id text not null default '',
  stem text not null,
  messages jsonb not null default '[]'::jsonb,
  updated_at timestamptz not null default now(),
  unique (user_id, doc_id, stem)
);

create index if not exists chat_history_user_stem_idx
  on public.chat_history (user_id, stem);

alter table public.chat_history enable row level security;

create policy "chat_history_select_own"
  on public.chat_history for select
  using (auth.uid() = user_id);

create policy "chat_history_insert_own"
  on public.chat_history for insert
  with check (auth.uid() = user_id);

create policy "chat_history_update_own"
  on public.chat_history for update
  using (auth.uid() = user_id)
  with check (auth.uid() = user_id);

create policy "chat_history_delete_own"
  on public.chat_history for delete
  using (auth.uid() = user_id);
