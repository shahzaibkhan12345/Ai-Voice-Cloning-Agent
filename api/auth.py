import jwt
from datetime import datetime, timedelta, timezone
from typing import Optional
from fastapi import HTTPException, Security, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from config import Config
from api.utils import get_logger

logger = get_logger(__name__)

# --- Token-based Security Scheme ---
# This defines an HTTP Bearer token scheme for FastAPI
security_scheme = HTTPBearer(auto_error=False) # auto_error=False to handle errors manually

# --- JWT Token Functions ---
def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """
    Creates a JWT access token.

    Args:
        data: The payload to encode into the token (e.g., {"sub": "user_id"}).
        expires_delta: Optional timedelta for token expiration. If None, uses default from Config.

    Returns:
        The encoded JWT string.
    """
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=Config.AUTH_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire, "iat": datetime.now(timezone.utc)})
    
    encoded_jwt = jwt.encode(
        to_encode, 
        Config.API_SECRET_KEY, 
        algorithm="HS256"
    )
    logger.debug(f"Access token created, expires at: {expire}")
    return encoded_jwt

def decode_access_token(token: str) -> Optional[dict]:
    """
    Decodes and validates a JWT access token.

    Args:
        token: The JWT string to decode.

    Returns:
        The decoded payload if valid, otherwise None.
    """
    try:
        # Use the same secret key and algorithm for decoding
        decoded_payload = jwt.decode(
            token,
            Config.API_SECRET_KEY,
            algorithms=["HS256"]
        )
        logger.debug(f"Token decoded successfully: {decoded_payload}")
        return decoded_payload
    except jwt.ExpiredSignatureError:
        logger.warning("Attempted to use an expired token.")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except jwt.InvalidTokenError as e:
        logger.error(f"Invalid token encountered: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except Exception as e:
        logger.exception(f"An unexpected error occurred during token decoding: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during token validation",
            headers={"WWW-Authenticate": "Bearer"},
        )

# --- FastAPI Dependency for Authentication ---
async def verify_token(credentials: Optional[HTTPAuthorizationCredentials] = Security(security_scheme)) -> dict:
    """
    FastAPI dependency to verify an incoming JWT token from the Authorization header.

    Args:
        credentials: The HTTPAuthorizationCredentials object provided by FastAPI's Security.

    Returns:
        The decoded token payload if authentication is successful.

    Raises:
        HTTPException: If no token is provided, token is invalid, or token has expired.
    """
    if not credentials or not credentials.credentials:
        logger.warning("No authentication token provided.")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    token = credentials.credentials
    payload = decode_access_token(token)
    
    if payload is None:
        # decode_access_token already raises HTTPException, but adding this for clarity
        logger.error("Token decoding failed, payload is None.")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    logger.info(f"User authenticated with token payload: {payload.get('sub', 'N/A')}")
    return payload

# --- Example Usage (for initial token generation, not part of API endpoint logic) ---
# In a real app, an /auth/login or /auth/token endpoint would provide this.
if __name__ == "__main__":
    # This block runs only when auth.py is executed directly
    print("Generating a sample token for user 'test_user'.")
    try:
        sample_token = create_access_token(data={"sub": "test_user"})
        print(f"Sample Token: {sample_token}")

        print("\nDecoding the sample token:")
        decoded_data = decode_access_token(sample_token)
        print(f"Decoded Data: {decoded_data}")

        # Simulate an expired token (for testing purposes, requires adjusting timedelta)
        # expired_token = create_access_token(data={"sub": "expired_user"}, expires_delta=timedelta(seconds=1))
        # import time
        # time.sleep(2)
        # decode_access_token(expired_token)

    except HTTPException as e:
        print(f"Error during token operation: {e.detail}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")