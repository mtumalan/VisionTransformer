# backend/users/serializers.py
from django.contrib.auth.models import User
from django.contrib.auth import authenticate
from rest_framework import serializers

class RegisterSerializer(serializers.ModelSerializer):
    """
    Serializer for registering a new user.
    Expects: username, password, email (optional), first_name, last_name (optional).
    """
    password = serializers.CharField(write_only=True, required=True, style={'input_type': 'password'})

    class Meta:
        model = User
        fields = ('username', 'password', 'email', 'first_name', 'last_name')

    def create(self, validated_data):
        # Use create_user so that the password is hashed
        user = User.objects.create_user(
            username=validated_data['username'],
            password=validated_data['password'],
            email=validated_data.get('email', ''),
        )
        return user

class LoginSerializer(serializers.Serializer):
    """
    Serializer for user login.
    Expects: username, password.
    """
    username = serializers.CharField(required=True)
    password = serializers.CharField(write_only=True, required=True, style={'input_type': 'password'})

    def validate(self, data):
        user = authenticate(username=data['username'], password=data['password'])
        if user is None:
            raise serializers.ValidationError('Invalid username or password.')
        if not user.is_active:
            raise serializers.ValidationError('User account is disabled.')
        data['user'] = user
        return data