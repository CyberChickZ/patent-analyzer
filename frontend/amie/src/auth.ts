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

export async function login(): Promise<User> {
  const result = await signInWithPopup(auth, provider);
  if (!result.user.email?.endsWith('@oregonstate.edu')) {
    await signOut(auth);
    throw new Error('Only @oregonstate.edu accounts are allowed');
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
