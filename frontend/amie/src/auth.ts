import { initializeApp } from 'firebase/app';
import { getAuth, signInWithPopup, GoogleAuthProvider, signOut, onAuthStateChanged, User } from 'firebase/auth';

const firebaseConfig = {
  apiKey: import.meta.env.VITE_FIREBASE_API_KEY || '',
  authDomain: import.meta.env.VITE_FIREBASE_AUTH_DOMAIN || '',
  projectId: import.meta.env.VITE_FIREBASE_PROJECT_ID || '',
};

const app = initializeApp(firebaseConfig);
const auth = getAuth(app);

const provider = new GoogleAuthProvider();
provider.setCustomParameters({ hd: 'oregonstate.edu' });

// Firebase hands back a code and nothing else. auth/unauthorized-domain is the
// one every deployment forgets: a new Cloud Run host has to be added to
// Firebase Auth's authorized domains by hand, and until it is, every sign-in
// fails with a string that names neither the fix nor who can apply it.
const AUTH_ERRORS: Record<string, string> = {
  'auth/unauthorized-domain': `This site is not authorized to sign you in: ${location.hostname} is missing from Firebase Auth's authorized domains. An administrator has to add it (Firebase console → Authentication → Settings → Authorized domains).`,
  'auth/operation-not-allowed': 'Google sign-in is turned off for this project. An administrator has to enable it in the Firebase console.',
  'auth/popup-blocked': 'Your browser blocked the sign-in window. Allow pop-ups for this site and try again.',
  'auth/popup-closed-by-user': 'The sign-in window was closed before sign-in finished.',
  'auth/network-request-failed': 'The sign-in request could not reach Google. Check your connection and try again.',
};

export function authErrorMessage(e: any): string {
  const code = e?.code || '';
  return AUTH_ERRORS[code] || e?.message || 'Sign-in failed.';
}

export async function login(): Promise<User> {
  let result;
  try {
    result = await signInWithPopup(auth, provider);
  } catch (e: any) {
    throw new Error(authErrorMessage(e));
  }
  if (!result.user.email?.endsWith('@oregonstate.edu')) {
    await signOut(auth);
    throw new Error(`Only @oregonstate.edu Google accounts can sign in. You signed in as ${result.user.email || 'an unknown account'}.`);
  }
  return result.user;
}

export async function logout(): Promise<void> {
  await signOut(auth);
}

export async function getToken(): Promise<string | null> {
  const user = auth.currentUser;
  if (!user) return null;
  return user.getIdToken();
}

export async function isDeveloper(): Promise<boolean> {
  const user = auth.currentUser;
  if (!user) return false;
  const result = await user.getIdTokenResult();
  return !!result.claims.developer;
}

export function onAuth(callback: (user: User | null) => void): void {
  onAuthStateChanged(auth, callback);
}

export function currentUser(): User | null {
  return auth.currentUser;
}
