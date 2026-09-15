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

// Firebase 只给错误码，不给人话。域名未授权是部署时最容易漏的一步：
// 新的 Cloud Run 前端域名必须手动加进 Firebase Auth 的 Authorized domains，
// 否则登录弹窗直接报 auth/unauthorized-domain，用户看到的只是一串码。
const AUTH_ERRORS: Record<string, string> = {
  'auth/unauthorized-domain': `域名未授权：${location.hostname} 不在 Firebase Auth 的授权域名列表里，请联系管理员在 Firebase 控制台 Authentication → Settings → Authorized domains 添加。`,
  'auth/operation-not-allowed': '登录方式未启用：请联系管理员在 Firebase 控制台开启 Google 登录。',
  'auth/popup-blocked': '登录弹窗被浏览器拦截了，请允许本站弹窗后重试。',
  'auth/popup-closed-by-user': '登录弹窗被关闭了，没有完成登录。',
  'auth/network-request-failed': '网络请求失败，请检查网络后重试。',
};

export function authErrorMessage(e: any): string {
  const code = e?.code || '';
  return AUTH_ERRORS[code] || e?.message || '登录失败';
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
    throw new Error(`只允许 @oregonstate.edu 的 Google 账号登录，当前账号是 ${result.user.email || '未知'}。`);
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
