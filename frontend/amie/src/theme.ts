export type Theme = "auto" | "light" | "dark";

const KEY = "amie_theme";

export function current(): Theme {
  try {
    const t = localStorage.getItem(KEY);
    if (t === "light" || t === "dark") return t;
  } catch { /* private mode */ }
  return "auto";
}

export function apply(t: Theme): void {
  if (t === "auto") document.documentElement.removeAttribute("data-theme");
  else document.documentElement.setAttribute("data-theme", t);
  try {
    if (t === "auto") localStorage.removeItem(KEY);
    else localStorage.setItem(KEY, t);
  } catch { /* private mode */ }
}

export function cycle(): Theme {
  const order: Theme[] = ["auto", "light", "dark"];
  const next = order[(order.indexOf(current()) + 1) % order.length];
  apply(next);
  return next;
}

export function glyph(t: Theme = current()): string {
  return t === "light" ? "☀ Light" : t === "dark" ? "☾ Dark" : "◐ Auto";
}
