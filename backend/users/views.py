from rest_framework import status
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import AllowAny, IsAuthenticated
from rest_framework.response import Response
from django.contrib.auth.models import User
from django.contrib.auth import authenticate
from .models import UserProfile
from django.http import JsonResponse
from mongoengine.errors import DoesNotExist, NotUniqueError
import logging

import json

from django.views.decorators.csrf import csrf_exempt
from .serializers import UserSerializer, UserProfileSerializer
from rest_framework_simplejwt.tokens import RefreshToken
from drf_yasg.utils import swagger_auto_schema
from drf_yasg import openapi
from django.contrib.auth.hashers import make_password
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


@swagger_auto_schema(
    method='get',
    responses={
        200: openapi.Response('Token is valid.'),
        401: openapi.Response('Unauthorized. Token is invalid or expired.'),
    },
)
@api_view(['GET'])
@permission_classes([IsAuthenticated])  # Only authenticated users can access this
def validate_token(request):
    """
    Validates the token and returns a response indicating whether the token is valid.

    :param request: Request object
    :return: Response object
    """
    # If the token is valid, this view will automatically be called.
    # No additional checks are necessary because IsAuthenticated handles token validation.
    return Response({"message": "Token is valid."}, status=status.HTTP_200_OK)


@swagger_auto_schema(
    method='post',
    request_body=openapi.Schema(
        type=openapi.TYPE_OBJECT,
        properties={
            'username': openapi.Schema(type=openapi.TYPE_STRING, description='Username for registration'),
            'password': openapi.Schema(type=openapi.TYPE_STRING, description='Password for registration'),
            'email': openapi.Schema(type=openapi.TYPE_STRING, description='Email address for registration'),
        },
        required=['username', 'password', 'email'],
    ),
    responses={
        201: openapi.Response('User created successfully.'),
        400: openapi.Response('All fields are required.'),
        401: openapi.Response('Unauthorized.'),
        404: openapi.Response('URL not found.'),
        500: openapi.Response('Internal server error.'),
    },
)
@api_view(['POST'])
@permission_classes([AllowAny])
def register(request):
    """
    Registers a new user with the given username, password, and email.
    """
    try:
        username = request.data.get('username')
        password = request.data.get('password')
        email = request.data.get('email')

        logger.debug(f"Starting registration for user: {username}")
        logger.debug(f"Email: {email}")

        if not username or not password or not email:
            return Response({"error": "All fields are required."}, status=status.HTTP_400_BAD_REQUEST)

        # Check if username already exists in Django auth
        if User.objects.filter(username=username).exists():
            return Response({"error": "Username already exists."}, status=status.HTTP_400_BAD_REQUEST)

        # Check if email already exists
        if User.objects.filter(email=email).exists():
            return Response({"error": "Email already exists."}, status=status.HTTP_400_BAD_REQUEST)

        # Create Django user
        user = User.objects.create_user(
            username=username,
            email=email,
            password=password
        )
        logger.debug(f"Created Django user: {username}")

        # Create MongoDB user profile
        try:
            user_profile = UserProfile(
                username=username,
                is_active=True
            ).save()
            logger.debug(f"Created MongoDB profile for user: {username}")
        except NotUniqueError:
            user.delete()  # Rollback Django user creation
            return Response({"error": "Username already exists in MongoDB."}, status=status.HTTP_400_BAD_REQUEST)
        except Exception as e:
            user.delete()  # Rollback Django user creation
            logger.error(f"Error creating MongoDB profile: {str(e)}")
            return Response({"error": "Failed to create user profile."}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        # Generate tokens
        refresh = RefreshToken.for_user(user)
        tokens = {
            'refresh': str(refresh),
            'access': str(refresh.access_token),
        }

        # Return success response with tokens and user data
        return Response({
            "message": "User registered successfully.",
            "tokens": tokens,
            "user": {
                "username": username,
                "email": email
            }
        }, status=status.HTTP_201_CREATED)

    except Exception as e:
        logger.error(f"Error in registration: {str(e)}")
        return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@swagger_auto_schema(
    method='post',
    request_body=openapi.Schema(
        type=openapi.TYPE_OBJECT,
        properties={
            'username': openapi.Schema(type=openapi.TYPE_STRING, description='Username for login'),
            'password': openapi.Schema(type=openapi.TYPE_STRING, description='Password for login'),
        },
        required=['username', 'password'],
    ),
    responses={
        200: openapi.Response('Login successful.'),
        401: openapi.Response('Invalid credentials.'),
        404: openapi.Response('User not found.'),
        500: openapi.Response('Internal server error.'),
    },
)
@api_view(['POST'])
@permission_classes([AllowAny])
def login(request):
    """
    Authenticates a user and returns JWT tokens.
    """
    try:
        username = request.data.get('username')
        password = request.data.get('password')

        logger.debug(f"Login attempt for user: {username}")
        logger.debug(f"Request data: {request.data}")

        if not username or not password:
            logger.warning("Missing username or password")
            return Response({"error": "Both username and password are required."}, status=status.HTTP_400_BAD_REQUEST)

        # Check if user exists first
        try:
            user_exists = User.objects.filter(username=username).exists()
            logger.debug(f"User exists check: {user_exists}")
            if not user_exists:
                logger.warning(f"User does not exist: {username}")
                return Response({"error": "Invalid credentials."}, status=status.HTTP_401_UNAUTHORIZED)
        except Exception as e:
            logger.error(f"Error checking user existence: {str(e)}")

        # Authenticate user
        user = authenticate(username=username, password=password)
        logger.debug(f"Authentication result: {'Success' if user else 'Failed'}")
        
        if not user:
            logger.warning(f"Failed login attempt for user: {username}")
            return Response({"error": "Invalid credentials."}, status=status.HTTP_401_UNAUTHORIZED)

        # Check if user profile exists in MongoDB
        try:
            user_profile = UserProfile.objects.get(username=username)
            user_profile.last_login = datetime.utcnow()
            user_profile.save()
            logger.debug(f"Updated last login for user: {username}")
        except DoesNotExist:
            # Create profile if it doesn't exist
            user_profile = UserProfile(
                username=username,
                is_active=True
            ).save()
            logger.debug(f"Created missing MongoDB profile for user: {username}")

        # Generate tokens
        refresh = RefreshToken.for_user(user)
        tokens = {
            'refresh': str(refresh),
            'access': str(refresh.access_token),
        }

        logger.debug(f"Login successful for user: {username}")
        return Response({
            "message": "Login successful.",
            "tokens": tokens,
            "user": {
                "username": username,
                "email": user.email
            }
        }, status=status.HTTP_200_OK)

    except Exception as e:
        logger.error(f"Error in login: {str(e)}")
        return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@swagger_auto_schema(
    method='post',
    request_body=openapi.Schema(
        type=openapi.TYPE_OBJECT,
        properties={
            'username': openapi.Schema(type=openapi.TYPE_STRING, description='Username to verify'),
            'email': openapi.Schema(type=openapi.TYPE_STRING, description='Email to verify'),
        },
        required=['username', 'email'],
    ),
    responses={
        200: openapi.Response('Username and email combination verified.'),
        404: openapi.Response('User not found.'),
        500: openapi.Response('Internal server error.'),
    },
)
@api_view(['POST'])
@permission_classes([AllowAny])
def verify_username_email(request):
    """
    Verifies if a combination of username and email exists in the User model.

    :param request: Request object
    :return: Response object
    """
    username = request.data.get('username')
    email = request.data.get('email')

    if not username or not email:
        return Response({"error": "Username and email are required."}, status=status.HTTP_400_BAD_REQUEST)

    try:
        # Find the user in the Django User model (not in UserProfile)
        user = User.objects.get(username=username, email=email)
        return Response({"message": "Username and email combination verified."}, status=status.HTTP_200_OK)
    except User.DoesNotExist:
        return Response({"error": "User not found."}, status=status.HTTP_404_NOT_FOUND)
    except Exception as e:
        return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@swagger_auto_schema(
    method='post',
    request_body=openapi.Schema(
        type=openapi.TYPE_OBJECT,
        properties={
            'username': openapi.Schema(type=openapi.TYPE_STRING, description='Username to reset password for'),
            'new_password': openapi.Schema(type=openapi.TYPE_STRING, description='New password'),
        },
        required=['username', 'new_password'],
    ),
    responses={
        200: openapi.Response('Password reset successfully.'),
        404: openapi.Response('User not found.'),
        400: openapi.Response('Bad request.'),
        500: openapi.Response('Internal server error.'),
    },
)
@api_view(['POST'])
@permission_classes([AllowAny])
@csrf_exempt
def reset_password(request):
    """
    Resets the password for the given username.

    :param request: Request object
    :return: Response object
    """
    username = request.data.get('username')
    new_password = request.data.get('new_password')

    if not username or not new_password:
        return Response({"error": "Username and new password are required."}, status=status.HTTP_400_BAD_REQUEST)

    try:
        # Find the user in the Django User model
        user = User.objects.get(username=username)

        # Update password (Django's set_password method hashes the password before saving)
        user.set_password(new_password)
        user.save()

        return Response({"message": "Password reset successfully."}, status=status.HTTP_200_OK)
    except User.DoesNotExist:
        return Response({"error": "User not found."}, status=status.HTTP_404_NOT_FOUND)
    except Exception as e:
        return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@swagger_auto_schema(
    method='get',
    responses={
        200: openapi.Response('User profile retrieved successfully.'),
        401: openapi.Response('Unauthorized.'),
        404: openapi.Response('User profile not found.'),
        500: openapi.Response('Internal server error.'),
    },
)
@api_view(['GET'])
@permission_classes([IsAuthenticated])
def user_profile(request):
    """
    Retrieves the user profile for the authenticated user.
    """
    try:
        username = request.user.username
        logger.debug(f"Fetching profile for user: {username}")

        # Get Django user
        try:
            django_user = User.objects.get(username=username)
        except User.DoesNotExist:
            logger.error(f"Django user not found: {username}")
            return Response({"error": "User not found"}, status=status.HTTP_404_NOT_FOUND)

        # Get MongoDB profile
        try:
            user_profile = UserProfile.objects.get(username=username)
        except DoesNotExist:
            logger.warning(f"MongoDB profile not found for user: {username}")
            # Create profile if it doesn't exist
            user_profile = UserProfile(
                username=username,
                is_active=True
            ).save()
            logger.debug(f"Created new MongoDB profile for user: {username}")

        # Get recent data
        recent_moods = user_profile.mood_history[-5:] if user_profile.mood_history else []
        recent_tracks = user_profile.listening_history[-5:] if user_profile.listening_history else []
        recent_recommendations = user_profile.recommendations[-5:] if user_profile.recommendations else []

        profile_data = {
            "id": str(user_profile.id),  # Convert ObjectId to string
            "username": username,
            "email": django_user.email,
            "created_at": user_profile.created_at,
            "last_login": user_profile.last_login,
            "recent_moods": recent_moods,
            "recent_tracks": recent_tracks,
            "recent_recommendations": recent_recommendations,
            "total_moods": len(user_profile.mood_history),
            "total_tracks": len(user_profile.listening_history),
            "total_recommendations": len(user_profile.recommendations)
        }

        logger.debug(f"Returning profile data: {profile_data}")
        return Response(profile_data, status=status.HTTP_200_OK)

    except Exception as e:
        logger.error(f"Error fetching user profile: {str(e)}")
        return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@swagger_auto_schema(
    method='put',
    request_body=openapi.Schema(
        type=openapi.TYPE_OBJECT,
        properties={},
    ),
    responses={
        200: openapi.Response('Profile updated successfully.'),
        401: openapi.Response('Unauthorized.'),
        404: openapi.Response('URL not found.'),
        500: openapi.Response('Internal server error.'),
    },
)
@api_view(['PUT'])
@permission_classes([IsAuthenticated])
def user_profile_update(request):
    """
    Update the user profile based on the request data.

    :param request: Request object
    :return: Response object
    """
    user = request.user
    profile = UserProfile.objects.get(username=user.username)

    # Update profile fields based on request data
    profile.save()
    return Response({"message": "Profile updated successfully."})


@swagger_auto_schema(
    method='delete',
    responses={
        200: openapi.Response('Profile deleted successfully.'),
        401: openapi.Response('Unauthorized.'),
        404: openapi.Response('URL not found.'),
        500: openapi.Response('Internal server error.'),
    },
)
@api_view(['DELETE'])
@permission_classes([IsAuthenticated])
def user_profile_delete(request):
    """
    Delete the user profile associated with the authenticated user.

    :param request: Request object
    :return: Response object
    """
    user = request.user
    profile = UserProfile.objects.get(username=user.username)
    profile.delete()
    return Response({"message": "Profile deleted successfully."})


@swagger_auto_schema(
    method='post',
    request_body=openapi.Schema(
        type=openapi.TYPE_OBJECT,
        properties={
            'recommendations': openapi.Schema(
                type=openapi.TYPE_ARRAY,
                items=openapi.Schema(
                    type=openapi.TYPE_OBJECT,
                    properties={
                        'name': openapi.Schema(type=openapi.TYPE_STRING, description='Name of the song'),
                        'artist': openapi.Schema(type=openapi.TYPE_STRING, description='Name of the artist'),
                        'preview_url': openapi.Schema(type=openapi.TYPE_STRING, nullable=True, description='Preview URL of the song'),
                        'external_url': openapi.Schema(type=openapi.TYPE_STRING, description='External URL of the song'),
                    },
                ),
                description='List of music recommendations'
            ),
        },
        required=['recommendations'],
    ),
    responses={
        201: openapi.Response('Recommendations saved successfully.'),
        400: openapi.Response('Recommendations are required.'),
        404: openapi.Response('User not found.'),
        500: openapi.Response('Internal server error.'),
    },
)
@api_view(['POST'])
@permission_classes([IsAuthenticated])
def save_recommendations(request, user_id):
    """
    Save music recommendations for the user with the given user ID.

    :param request: The request object containing the music recommendations.
    :param user_id: The user ID to save the recommendations for.
    :return: The response object indicating the status of the operation.
    """
    try:
        recommendations = request.data.get("recommendations")

        if not recommendations:
            return Response({"error": "Recommendations are required"}, status=status.HTTP_400_BAD_REQUEST)

        user_profile = UserProfile.objects(id=user_id).first()
        if not user_profile:
            return Response({"error": "User not found"}, status=status.HTTP_404_NOT_FOUND)

        user_profile.recommendations.extend(recommendations)
        user_profile.save()

        return Response({"message": "Recommendations saved successfully"}, status=status.HTTP_201_CREATED)
    except Exception as e:
        return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@swagger_auto_schema(
    method='get',
    responses={
        200: openapi.Response('Recommendations retrieved successfully.'),
        404: openapi.Response('User not found.'),
        500: openapi.Response('Internal server error.'),
    },
)
@api_view(['GET'])
@permission_classes([IsAuthenticated])
def get_recommendations(request, user_id):
    """
    Retrieve music recommendations for the user with the given user ID.

    :param request: The request object containing the user ID.
    :param user_id: The user ID to retrieve recommendations for.
    :return: The response object containing the music recommendations.
    """
    try:
        # Retrieve user profile using MongoDB ObjectId
        user_profile = UserProfile.objects(id=user_id).first()

        if not user_profile:
            return Response({"error": "User not found"}, status=status.HTTP_404_NOT_FOUND)

        return Response({"recommendations": user_profile.recommendations}, status=status.HTTP_200_OK)

    except Exception as e:
        return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@swagger_auto_schema(
    method='delete',
    manual_parameters=[
        openapi.Parameter('user_id', openapi.IN_PATH, description='User ID to delete recommendations for', type=openapi.TYPE_STRING),
    ],
    responses={
        200: openapi.Response('All recommendations deleted successfully.'),
        401: openapi.Response('Unauthorized.'),
        404: openapi.Response('User not found.'),
        500: openapi.Response('Internal server error.'),
    },
)
@api_view(['DELETE'])
def delete_all_recommendations(request, user_id):
    """
    Delete all music recommendations for the user with the given user ID.

    :param request: The request object containing the user ID.
    :param user_id: The user ID to delete recommendations for.
    :return: The response object indicating the status of the operation.
    """
    try:
        # Retrieve user profile by MongoDB ObjectId
        user_profile = UserProfile.objects(id=user_id).first()

        if not user_profile:
            return Response({"error": "User not found"}, status=status.HTTP_404_NOT_FOUND)

        # Explicitly set recommendations to an empty list
        user_profile.recommendations = []
        user_profile.save()

        return Response({"message": "All recommendations deleted"}, status=status.HTTP_200_OK)

    except Exception as e:
        return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@csrf_exempt
@swagger_auto_schema(
    method='post',
    request_body=openapi.Schema(
        type=openapi.TYPE_OBJECT,
        properties={
            'recommendations': openapi.Schema(
                type=openapi.TYPE_ARRAY,
                items=openapi.Schema(
                    type=openapi.TYPE_OBJECT,
                    properties={
                        'name': openapi.Schema(type=openapi.TYPE_STRING),
                        'artist': openapi.Schema(type=openapi.TYPE_STRING),
                        'preview_url': openapi.Schema(type=openapi.TYPE_STRING, nullable=True),
                        'external_url': openapi.Schema(type=openapi.TYPE_STRING),
                    },
                ),
            ),
        },
        required=['recommendations'],
    ),
    responses={
        201: openapi.Response('Recommendations saved successfully.'),
        400: openapi.Response('Recommendations are required.'),
        401: openapi.Response('Unauthorized.'),
        404: openapi.Response('User not found.'),
        500: openapi.Response('Internal server error.'),
    },
)
@api_view(['POST', 'GET', 'DELETE'])
def user_recommendations(request, user_id):
    """
    This function allows users to save, retrieve, and delete music recommendations.

    :param request: The request object containing the music recommendations.
    :param user_id: The user ID to save the recommendations for.
    :return: The response object indicating the status of the operation.
    """
    if request.method == 'POST':
        data = json.loads(request.body)
        try:
            user_profile = UserProfile.objects.get(id=user_id)
            user_profile.recommendations.extend(data.get('recommendations', []))
            user_profile.save()
            return Response({"message": "Recommendations saved successfully."}, status=status.HTTP_201_CREATED)
        except DoesNotExist:
            return Response({"error": "User not found."}, status=status.HTTP_404_NOT_FOUND)

    elif request.method == 'GET':
        try:
            user_profile = UserProfile.objects.get(id=user_id)
            return Response({"recommendations": user_profile.recommendations}, status=status.HTTP_200_OK)
        except DoesNotExist:
            return Response({"error": "User not found."}, status=status.HTTP_404_NOT_FOUND)

    elif request.method == 'DELETE':
        try:
            user_profile = UserProfile.objects.get(id=user_id)
            user_profile.recommendations.clear()  # Clear all recommendations
            user_profile.save()
            return Response({"message": "All recommendations deleted."}, status=status.HTTP_204_NO_CONTENT)
        except DoesNotExist:
            return Response({"error": "User not found."}, status=status.HTTP_404_NOT_FOUND)


@swagger_auto_schema(
    method='get',
    responses={
        200: openapi.Response('User mood history retrieved successfully.'),
        401: openapi.Response('Unauthorized.'),
        404: openapi.Response('User not found.'),
        500: openapi.Response('Internal server error.'),
    },
)
@api_view(['GET'])
@permission_classes([IsAuthenticated])
def user_mood_history(request):
    """
    Retrieves the mood history for the authenticated user.
    """
    try:
        username = request.user.username
        logger.debug(f"Fetching mood history for user: {username}")

        try:
            user_profile = UserProfile.objects.get(username=username)
        except DoesNotExist:
            logger.error(f"User profile not found: {username}")
            return Response({"error": "User profile not found"}, status=status.HTTP_404_NOT_FOUND)

        # Get all mood history, sorted by timestamp
        mood_history = sorted(
            user_profile.mood_history,
            key=lambda x: x['timestamp'],
            reverse=True
        )

        return Response({
            "mood_history": mood_history,
            "total_entries": len(mood_history)
        }, status=status.HTTP_200_OK)

    except Exception as e:
        logger.error(f"Error fetching mood history: {str(e)}")
        return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@swagger_auto_schema(
    method='get',
    responses={
        200: openapi.Response('User listening history retrieved successfully.'),
        401: openapi.Response('Unauthorized.'),
        404: openapi.Response('User not found.'),
        500: openapi.Response('Internal server error.'),
    },
)
@api_view(['GET'])
@permission_classes([IsAuthenticated])
def user_listening_history(request):
    """
    Retrieves the listening history for the authenticated user.
    """
    try:
        username = request.user.username
        logger.debug(f"Fetching listening history for user: {username}")

        try:
            user_profile = UserProfile.objects.get(username=username)
        except DoesNotExist:
            logger.error(f"User profile not found: {username}")
            return Response({"error": "User profile not found"}, status=status.HTTP_404_NOT_FOUND)

        # Get all listening history, sorted by timestamp
        listening_history = sorted(
            user_profile.listening_history,
            key=lambda x: x['timestamp'],
            reverse=True
        )

        return Response({
            "listening_history": listening_history,
            "total_entries": len(listening_history)
        }, status=status.HTTP_200_OK)

    except Exception as e:
        logger.error(f"Error fetching listening history: {str(e)}")
        return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
