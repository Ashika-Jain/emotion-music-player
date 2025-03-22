import React from "react";
import { Box, Typography, Paper, Button } from "@mui/material";
import { Link } from "react-router-dom";
import { useLocation } from "react-router-dom";
import axios from "axios";

const API_URL = "http://localhost:8000";

// Get the user token from localStorage
const getToken = () => {
  const user = JSON.parse(localStorage.getItem("user"));
  return user?.access;
};

// Create axios instance with authorization header
const createAuthAxios = () => {
  const token = getToken();
  return axios.create({
    baseURL: API_URL,
    headers: {
      Authorization: token ? `Bearer ${token}` : "",
    },
  });
};

export const detectTextEmotion = async (text) => {
  try {
    const authAxios = createAuthAxios();
    const response = await authAxios.post(`${API_URL}/api/text_emotion/`, { text });
    return response.data;
  } catch (error) {
    console.error("Text emotion detection error:", error);
    throw error;
  }
};

export const detectFacialEmotion = async (imageFile) => {
  try {
    const authAxios = createAuthAxios();
    const formData = new FormData();
    formData.append("image", imageFile);
    
    const response = await authAxios.post(`${API_URL}/api/facial_emotion/`, formData, {
      headers: {
        "Content-Type": "multipart/form-data",
      },
    });
    
    return response.data;
  } catch (error) {
    console.error("Facial emotion detection error:", error);
    throw error;
  }
};

const Emotion = () => {
  const location = useLocation();
  const { emotion } = location.state || { emotion: "" };

  return (
    <Box style={styles.container}>
      <Typography variant="h5" style={styles.title}>
        <strong>
          Detected Mood: <span style={styles.emotion}>{emotion}</span>
        </strong>
      </Typography>
      <Paper elevation={4} style={styles.paper}>
        <Typography variant="h6" style={styles.subtitle}>
          How are you feeling today?
        </Typography>
        <Box style={styles.buttonContainer}>
          <Link to="/results" style={styles.link}>
            <Button variant="contained" color="primary" style={styles.button}>
              Get Recommendations
            </Button>
          </Link>
        </Box>
      </Paper>
    </Box>
  );
};
