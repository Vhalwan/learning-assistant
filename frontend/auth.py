"""Supabase email/password auth gate for the Streamlit app."""
from __future__ import annotations

import time
from typing import Any, Dict

import streamlit as st
import streamlit.components.v1 as components

from backend.supabase_client import (
    apply_session,
    get_supabase_client,
    session_to_dict,
    user_to_dict,
)
from backend.user_context import set_user_id

SESSION_USER_KEY = "auth_user"
SESSION_AUTH_KEY = "auth_session"
FORGOT_PASSWORD_KEY = "show_forgot_password"
RECOVERY_SESSION_KEY = "password_recovery_session"


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
    keys_to_clear = [
        k for k in list(st.session_state.keys())
        if k not in ("use_api_mode", "api_token", "use_faiss_search")
    ]
    for k in keys_to_clear:
        del st.session_state[k]


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


def _query_param(name: str) -> str:
    value = st.query_params.get(name)
    if isinstance(value, list):
        value = value[0] if value else ""
    return str(value or "")


def _inject_recovery_hash_detector() -> None:
    components.html(
        """
        <script>
        (function () {
          const parentWindow = window.parent;
          if (!parentWindow || !parentWindow.location) return;

          const hash = parentWindow.location.hash || "";
          if (!hash || !hash.includes("type=recovery")) return;

          const hashParams = new URLSearchParams(hash.replace(/^#/, ""));
          if (hashParams.get("type") !== "recovery" || !hashParams.get("access_token")) {
            return;
          }

          const nextUrl = new URL(parentWindow.location.href);
          nextUrl.hash = "";
          [
            "type",
            "access_token",
            "refresh_token",
            "expires_at",
            "expires_in",
            "token_type"
          ].forEach((key) => {
            const value = hashParams.get(key);
            if (value) nextUrl.searchParams.set(key, value);
          });

          parentWindow.location.replace(nextUrl.toString());
        })();
        </script>
        """,
        height=0,
        width=0,
    )


def _sync_recovery_session_from_query() -> None:
    if _query_param("type") != "recovery":
        return

    access_token = _query_param("access_token")
    if not access_token:
        return

    st.session_state[RECOVERY_SESSION_KEY] = {
        "access_token": access_token,
        "refresh_token": _query_param("refresh_token"),
    }
    st.session_state[FORGOT_PASSWORD_KEY] = False
    st.query_params.clear()
    st.rerun()


def _render_recovery_screen() -> None:
    st.markdown(
        """
        <style>
          .stApp {
            background: #f8fafc;
          }

          div[data-testid="stMainBlockContainer"] {
            max-width: 1040px;
            padding-top: 4.5rem;
          }

          .lectova-auth-brand {
            text-align: center;
            margin: 0 auto 1.65rem;
          }

          .lectova-auth-name {
            color: #0f172a;
            font-size: clamp(2.7rem, 6vw, 4.25rem);
            font-weight: 800;
            line-height: 1;
            letter-spacing: 0;
          }

          .lectova-auth-subtitle {
            color: #64748b;
            font-size: 1.06rem;
            font-weight: 500;
            margin-top: 0.7rem;
          }

          div[data-testid="stVerticalBlockBorderWrapper"] {
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 8px;
            box-shadow: 0 18px 50px rgba(15, 23, 42, 0.09);
            padding: 1.35rem 1.45rem 1.5rem;
          }

          div[data-testid="stTextInput"] {
            margin-bottom: 0.72rem;
          }

          div[data-testid="stTextInput"] label {
            color: #334155;
            font-weight: 650;
            padding-bottom: 0.25rem;
          }

          div[data-testid="stTextInput"] input {
            border: 1px solid #cbd5e1;
            border-radius: 8px;
            min-height: 2.85rem;
            padding: 0.75rem 0.9rem;
            background: #ffffff;
            color: #0f172a;
          }

          div[data-testid="stTextInput"] input:focus {
            border-color: #0f766e;
            box-shadow: 0 0 0 3px rgba(15, 118, 110, 0.14);
          }

          .st-key-auth_new_password input::-ms-reveal,
          .st-key-auth_new_password input::-ms-clear,
          .st-key-auth_confirm_password input::-ms-reveal,
          .st-key-auth_confirm_password input::-ms-clear {
            display: none !important;
            width: 0 !important;
            height: 0 !important;
          }

          .st-key-auth_new_password [data-testid="InputInstructions"],
          .st-key-auth_confirm_password [data-testid="InputInstructions"] {
            display: none !important;
          }

          div[data-testid="stButton"] > button {
            border-radius: 8px;
            min-height: 2.85rem;
            font-weight: 750;
            margin-top: 0.25rem;
          }

          div[data-testid="stButton"] > button[kind="primary"] {
            background: #0f766e;
            border-color: #0f766e;
          }

          div[data-testid="stButton"] > button[kind="primary"]:hover {
            background: #115e59;
            border-color: #115e59;
          }
        </style>

        <div class="lectova-auth-brand">
          <div class="lectova-auth-name">Lectova</div>
          <div class="lectova-auth-subtitle">Set new password</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    left_pad, form_col, right_pad = st.columns([1, 1.15, 1])
    with form_col:
        with st.container(border=True):
            new_password = st.text_input(
                "New password",
                type="password",
                key="auth_new_password",
                autocomplete="new-password",
            )
            confirm_password = st.text_input(
                "Confirm password",
                type="password",
                key="auth_confirm_password",
                autocomplete="new-password",
            )
            update_clicked = st.button(
                "Update password",
                type="primary",
                use_container_width=True,
                key="auth_update_password_submit",
            )

            if not update_clicked:
                return

            if new_password != confirm_password:
                st.error("Passwords do not match")
                return
            if not new_password:
                st.error("Enter a new password.")
                return

            recovery_session = st.session_state.get(RECOVERY_SESSION_KEY) or {}
            try:
                client = get_supabase_client()
                apply_session(
                    recovery_session["access_token"],
                    recovery_session.get("refresh_token", ""),
                )
                client.auth.update_user({"password": new_password})
                st.success("Password updated — you can now log in")
                st.session_state.pop(RECOVERY_SESSION_KEY, None)
                _clear_auth_state()
                time.sleep(2)
                st.rerun()
            except Exception as exc:
                st.error(str(exc))


def handle_password_recovery() -> None:
    """Show the reset-completion form when Supabase returns a recovery hash."""
    _inject_recovery_hash_detector()
    _sync_recovery_session_from_query()

    if st.session_state.get(RECOVERY_SESSION_KEY):
        _render_recovery_screen()
        st.stop()


def _sync_forgot_password_link_state() -> None:
    forgot_param = st.query_params.get("forgot_password")
    if isinstance(forgot_param, list):
        forgot_param = forgot_param[0] if forgot_param else None
    if forgot_param == "1":
        st.session_state[FORGOT_PASSWORD_KEY] = True
        st.query_params.clear()
    elif forgot_param == "0":
        st.session_state[FORGOT_PASSWORD_KEY] = False
        st.query_params.clear()


def render_auth_screen() -> None:
    _sync_forgot_password_link_state()

    st.markdown(
        """
        <style>
          .stApp {
            background: #f8fafc;
          }

          div[data-testid="stMainBlockContainer"] {
            max-width: 1040px;
            padding-top: 4.5rem;
          }

          .lectova-auth-brand {
            text-align: center;
            margin: 0 auto 1.65rem;
          }

          .lectova-auth-name {
            color: #0f172a;
            font-size: clamp(2.7rem, 6vw, 4.25rem);
            font-weight: 800;
            line-height: 1;
            letter-spacing: 0;
          }

          .lectova-auth-subtitle {
            color: #64748b;
            font-size: 1.06rem;
            font-weight: 500;
            margin-top: 0.7rem;
          }

          div[data-testid="stVerticalBlockBorderWrapper"] {
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 8px;
            box-shadow: 0 18px 50px rgba(15, 23, 42, 0.09);
            padding: 1.35rem 1.45rem 1.5rem;
          }

          div[data-testid="stTabs"] button[data-baseweb="tab"] {
            font-weight: 700;
            color: #475569;
            padding-top: 0.25rem;
            padding-bottom: 0.65rem;
          }

          div[data-testid="stTabs"] button[aria-selected="true"] {
            color: #0f766e;
          }

          div[data-testid="stTabs"] [data-baseweb="tab-highlight"] {
            background-color: #0f766e;
            height: 3px;
            border-radius: 2px;
          }

          div[data-testid="stTextInput"] {
            margin-bottom: 0.72rem;
          }

          div[data-testid="stTextInput"] label {
            color: #334155;
            font-weight: 650;
            padding-bottom: 0.25rem;
          }

          div[data-testid="stTextInput"] input {
            border: 1px solid #cbd5e1;
            border-radius: 8px;
            min-height: 2.85rem;
            padding: 0.75rem 0.9rem;
            background: #ffffff;
            color: #0f172a;
          }

          div[data-testid="stTextInput"] input:focus {
            border-color: #0f766e;
            box-shadow: 0 0 0 3px rgba(15, 118, 110, 0.14);
          }

          .st-key-auth_login_password input::-ms-reveal,
          .st-key-auth_login_password input::-ms-clear,
          .st-key-auth_signup_password input::-ms-reveal,
          .st-key-auth_signup_password input::-ms-clear {
            display: none !important;
            width: 0 !important;
            height: 0 !important;
          }

          .st-key-auth_login_password [data-testid="InputInstructions"],
          .st-key-auth_signup_password [data-testid="InputInstructions"] {
            display: none !important;
          }

          div[data-testid="stButton"] > button {
            border-radius: 8px;
            min-height: 2.85rem;
            font-weight: 750;
            margin-top: 0.25rem;
          }

          div[data-testid="stButton"] > button[kind="primary"] {
            background: #0f766e;
            border-color: #0f766e;
          }

          div[data-testid="stButton"] > button[kind="primary"]:hover {
            background: #115e59;
            border-color: #115e59;
          }

          .lectova-auth-link {
            color: #0f766e !important;
            font-size: 0.88rem;
            font-weight: 700;
            text-decoration: none !important;
          }

          .lectova-auth-link:hover {
            color: #115e59 !important;
            text-decoration: underline !important;
          }

          .lectova-forgot-link-row {
            margin: -0.25rem 0 0.85rem;
            text-align: right;
          }

          .lectova-back-link-row {
            margin-bottom: 1rem;
          }
        </style>

        <div class="lectova-auth-brand">
          <div class="lectova-auth-name">Lectova</div>
          <div class="lectova-auth-subtitle">Your AI-powered lecture companion</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    left_pad, form_col, right_pad = st.columns([1, 1.15, 1])
    with form_col:
        with st.container(border=True):
            if st.session_state.get(FORGOT_PASSWORD_KEY):
                st.markdown(
                    """
                    <div class="lectova-back-link-row">
                      <a class="lectova-auth-link" href="?forgot_password=0" target="_self">
                        &larr; Back to login
                      </a>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                reset_email = st.text_input(
                    "Email",
                    key="auth_reset_email",
                    autocomplete="email",
                    placeholder="you@example.com",
                )
                reset_clicked = st.button(
                    "Send reset email",
                    type="primary",
                    use_container_width=True,
                    key="auth_reset_submit",
                )

                if reset_clicked:
                    if not (reset_email or "").strip():
                        st.error("Enter your email.")
                    else:
                        try:
                            client = get_supabase_client()
                            client.auth.reset_password_email(reset_email.strip())
                            st.success("Check your email for a reset link")
                        except Exception as exc:
                            st.error(str(exc))
                return

            login_tab, signup_tab = st.tabs(["Login", "Sign Up"])

            with login_tab:
                email = st.text_input(
                    "Email",
                    key="auth_login_email",
                    autocomplete="email",
                    placeholder="you@example.com",
                )
                password = st.text_input(
                    "Password",
                    type="password",
                    key="auth_login_password",
                    autocomplete="current-password",
                    placeholder="Enter your password",
                )
                st.markdown(
                    """
                    <div class="lectova-forgot-link-row">
                      <a class="lectova-auth-link" href="?forgot_password=1" target="_self">
                        Forgot password?
                      </a>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                login_clicked = st.button(
                    "Login",
                    type="primary",
                    use_container_width=True,
                    key="auth_login_submit",
                )

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

            with signup_tab:
                register_email = st.text_input(
                    "Email",
                    key="auth_signup_email",
                    autocomplete="email",
                    placeholder="you@example.com",
                )
                register_password = st.text_input(
                    "Password",
                    type="password",
                    key="auth_signup_password",
                    autocomplete="new-password",
                    placeholder="Create a password",
                )
                register_clicked = st.button(
                    "Sign Up",
                    type="primary",
                    use_container_width=True,
                    key="auth_signup_submit",
                )

                if register_clicked:
                    if not (register_email or "").strip() or not register_password:
                        st.error("Enter your email and password.")
                    else:
                        try:
                            client = get_supabase_client()
                            res = client.auth.sign_up(
                                {
                                    "email": register_email.strip(),
                                    "password": register_password,
                                }
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
