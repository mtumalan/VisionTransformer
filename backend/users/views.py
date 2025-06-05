# backend/users/views.py
from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView
from django.contrib.auth import login

from .serializers import RegisterSerializer, LoginSerializer

class RegisterView(APIView):
    """
    POST /api/register/  →  create a new user.
    """

    permission_classes = []  # Allow any user (authenticated or not) to hit this endpoint

    def post(self, request, *args, **kwargs):
        serializer = RegisterSerializer(data=request.data)
        if serializer.is_valid():
            user = serializer.save()
            data = {
                'id': user.id,
                'username': user.username,
                'email': user.email,
            }
            return Response(data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

class LoginView(APIView):
    """
    POST /api/login/  →  authenticate and log in a user.
    """
    permission_classes = []

    def post(self, request):
        serializer = LoginSerializer(data=request.data)
        if serializer.is_valid():
            user = serializer.validated_data['user']
            login(request, user)
            data = {
                'id': user.id,
                'username': user.username,
                'email': user.email,
            }
            return Response(data, status=status.HTTP_200_OK)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
