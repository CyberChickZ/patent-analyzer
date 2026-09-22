#!/usr/bin/env python3
"""Mark a user as developer via Firebase Custom Claims.

Usage:
  python scripts/set_developer.py zhanhaoc@oregonstate.edu
  python scripts/set_developer.py zhanhaoc@oregonstate.edu --remove
"""
import sys
import firebase_admin
from firebase_admin import auth

firebase_admin.initialize_app()

email = sys.argv[1]
remove = "--remove" in sys.argv

user = auth.get_user_by_email(email)
claims = {} if remove else {"developer": True}
auth.set_custom_user_claims(user.uid, claims)
action = "Removed developer" if remove else "Set developer"
print(f"{action} for {email} (uid={user.uid})")
