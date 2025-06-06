from django.contrib.auth import authenticate, login as django_login
from rest_framework import status
from rest_framework.permissions import AllowAny, IsAuthenticated
from rest_framework.views import APIView
from rest_framework.response import Response
from .serializers import LoginSerializer, RegisterSerializer
from django.conf import settings
from django.contrib.auth import logout as django_logout


class LoginView(APIView):
    """
    POST { username, password } → authenticate + create a session (sessionid cookie).
    """
    permission_classes = [AllowAny]

    def post(self, request, *args, **kwargs):
        # 1) Validate credentials via the serializer
        serializer = LoginSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        user = serializer.validated_data["user"]

        # 2) Call Django's login() to set request.session and issue sessionid cookie
        django_login(request, user)

        # 3) Return some basic user info (or whatever you like)
        return Response(
            {
                "id": user.id,
                "username": user.username,
                "email": user.email,
            },
            status=status.HTTP_200_OK,
        )

class RegisterView(APIView):
    """
    POST { username, password, email, … } → create a new user.
    """
    permission_classes = [AllowAny]

    def post(self, request, *args, **kwargs):
        serializer = RegisterSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        user = serializer.save()
        return Response(
            {
                "id": user.id,
                "username": user.username,
                "email": user.email,
            },
            status=status.HTTP_201_CREATED,
        )

class CurrentUserView(APIView):
    """
    GET → return current user info if the sessionid cookie is valid.
    """
    permission_classes = [AllowAny]

    def get(self, request):
        user = request.user
        return Response({
            "id": user.id,
            "username": user.username,
            "email": user.email,
        })

class LogoutView(APIView):
    permission_classes = [AllowAny]

    def post(self, request):
        django_logout(request)            # destroys DB session + rotates key
        resp = Response(status=status.HTTP_204_NO_CONTENT)
        resp.delete_cookie(
            settings.SESSION_COOKIE_NAME,
            path="/",
            samesite=settings.SESSION_COOKIE_SAMESITE,
        )
        return resp