import axios from "axios";

const API_URL = "http://localhost:8000";

export const register = async (username, password, email) => {
  try {
    await axios.post(`${API_URL}/users/register/`, {
      username,
      password,
      email,
    });
  } catch (error) {
    console.error("Registration error:", error);
    throw error;
  }
};

export const login = async (username, password) => {
  try {
    const response = await axios.post(`${API_URL}/users/login/`, {
      username,
      password,
    });
    
    if (response.data.access) {
      localStorage.setItem("user", JSON.stringify(response.data));
    }
    
    return response.data;
  } catch (error) {
    console.error("Login error:", error);
    throw error;
  }
};
