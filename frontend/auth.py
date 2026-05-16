"""Supabase email/password auth gate for the Streamlit app."""
from __future__ import annotations

from typing import Any, Dict

import streamlit as st

from backend.supabase_client import (
    apply_session,
    get_supabase_client,
    session_to_dict,
    user_to_dict,
)
from backend.user_context import set_user_id

SESSION_USER_KEY = "auth_user"
SESSION_AUTH_KEY = "auth_session"


def _clear_auth_state() -> None:
    for key in (SESSION_USER_KEY, SESSION_AUTH_KEY, "current_stem", "current_pdf_filename"):
        st.session_state.pop(key, None)
    set_user_id(None)


def logout() -> None:
    try:
        get_supabase_client().auth.sign_out()
    except Exception:
        pass
    _clear_auth_state()


def _restore_session_from_state() -> bool:
    tokens = st.session_state.get(SESSION_AUTH_KEY)
    user = st.session_state.get(SESSION_USER_KEY)
    if not tokens or not user:
        return False
    try:
        apply_session(tokens["access_token"], tokens["refresh_token"])
        refreshed = get_supabase_client().auth.get_user()
        if not refreshed or not refreshed.user:
            _clear_auth_state()
            return False
        st.session_state[SESSION_USER_KEY] = user_to_dict(refreshed.user)
        set_user_id(st.session_state[SESSION_USER_KEY]["id"])
        return True
    except Exception:
        _clear_auth_state()
        return False


def _auth_error_message(exc: Exception, *, registering: bool) -> str:
    text = str(exc).lower()
    code = str(getattr(exc, "code", "") or "").lower()
    if "invalid api key" in text or code == "invalid_api_key":
        return (
            "Supabase rejected the API key. In .env use SUPABASE_ANON_KEY "
            "(the anon/public key from Project Settings → API), then restart the app."
        )
    if registering:
        if "already" in text and ("registered" in text or "exists" in text):
            return "An account with this email already exists. Try logging in instead."
        if "password" in text and ("short" in text or "least" in text or "weak" in text):
            return "Password is too weak. Use at least 6 characters."
        if "invalid" in text and "email" in text:
            return "Enter a valid email address."
        if "signup" in text and "disabled" in text:
            return "Sign-ups are disabled in Supabase. Enable email sign-up under Authentication → Providers."
        return f"Could not create account: {exc}"
    if "invalid" in text and ("credentials" in text or "login" in text):
        return "Wrong email or password."
    if "email not confirmed" in text:
        return "Confirm your email before signing in (check your inbox)."
    return f"Sign-in failed: {exc}"


def _persist_session(session: Any, user: Any) -> Dict[str, str]:
    st.session_state[SESSION_AUTH_KEY] = session_to_dict(session)
    st.session_state[SESSION_USER_KEY] = user_to_dict(user)
    apply_session(session.access_token, session.refresh_token)
    set_user_id(st.session_state[SESSION_USER_KEY]["id"])
    return st.session_state[SESSION_USER_KEY]


def render_auth_screen() -> None:
    st.markdown(
        """
        <div class="app-header" style="margin-bottom: 1.5rem;">
          <div class="app-logo">L</div>
          <div>
            <div class="app-title">Lectova</div>
            <div class="app-subtitle">Sign in to continue</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    email = st.text_input("Email", key="auth_email", autocomplete="email")
    password = st.text_input(
        "Password",
        type="password",
        key="auth_password",
        autocomplete="current-password",
    )

    col_login, col_register = st.columns(2)
    with col_login:
        login_clicked = st.button("Log in", type="primary", use_container_width=True)
    with col_register:
        register_clicked = st.button("Register", use_container_width=True)

    if login_clicked:
        if not (email or "").strip() or not password:
            st.error("Enter your email and password.")
        else:
            try:
                client = get_supabase_client()
                res = client.auth.sign_in_with_password(
                    {"email": email.strip(), "password": password}
                )
                if not res.session or not res.user:
                    st.error("Sign-in failed. Check your email and password.")
                else:
                    _persist_session(res.session, res.user)
                    st.rerun()
            except Exception as exc:
                st.error(_auth_error_message(exc, registering=False))

    if register_clicked:
        if not (email or "").strip() or not password:
            st.error("Enter your email and password.")
        else:
            try:
                client = get_supabase_client()
                res = client.auth.sign_up(
                    {"email": email.strip(), "password": password}
                )
                if res.session and res.user:
                    _persist_session(res.session, res.user)
                    st.rerun()
                else:
                    st.success(
                        "Account created. If email confirmation is enabled, "
                        "check your inbox, then log in."
                    )
            except Exception as exc:
                st.error(_auth_error_message(exc, registering=True))


def require_auth() -> Dict[str, str]:
    """Return the logged-in user dict, or show login and stop the app."""
    if _restore_session_from_state():
        return st.session_state[SESSION_USER_KEY]

    render_auth_screen()
    st.stop()
