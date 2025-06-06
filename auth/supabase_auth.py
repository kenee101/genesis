from supabase import create_client, Client
import streamlit as st
import os
from typing import Optional, Tuple


def init_supabase() -> Client:
    """Initialize Supabase client."""
    url = st.secrets.supabase.url
    key = st.secrets.supabase.key
    return create_client(url, key)


def sign_up(email: str, password: str) -> tuple[bool, str]:
    """Sign up a new user with email and password"""
    try:
        supabase = init_supabase()
        # Get the site URL from secrets or use default
        site_url = st.secrets.get("supabase", {}).get(
            "site_url", "http://localhost:8501")

        # Sign up the user with redirect URL
        response = supabase.auth.sign_up({
            "email": email,
            "password": password,
            "options": {
                "email_redirect_to": f"{site_url}/?auth=success"
            }
        })

        if response.user:
            return True, "Registration successful! Please check your email to confirm your account."
        return False, "Registration failed. Please try again."
    except Exception as e:
        return False, f"Error during registration: {str(e)}"


def sign_in(email: str, password: str) -> Tuple[bool, str]:
    """Sign in a user with Supabase."""
    try:
        supabase = init_supabase()
        response = supabase.auth.sign_in_with_password({
            "email": email,
            "password": password
        })
        return True, "Login successful!"
    except Exception as e:
        return False, str(e)


def sign_out() -> None:
    """Sign out the current user."""
    try:
        supabase = init_supabase()
        supabase.auth.sign_out()
    except Exception as e:
        st.error(f"Error signing out: {str(e)}")


def get_current_user() -> Optional[dict]:
    """Get the current user's session."""
    try:
        supabase = init_supabase()
        user = supabase.auth.get_user()
        return user
    except Exception:
        return None


def reset_password(email: str) -> Tuple[bool, str]:
    """Send password reset email."""
    try:
        supabase = init_supabase()
        response = supabase.auth.reset_password_email(email)
        return True, "Password reset email sent!"
    except Exception as e:
        return False, str(e)
