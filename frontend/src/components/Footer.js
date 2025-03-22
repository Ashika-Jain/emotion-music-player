import React from "react";
import { Box } from "@mui/material";

const Footer = () => {
  return (
    <Box sx={styles.footer}></Box>
  );
};

const styles = {
  footer: {
    backgroundColor: "#ff4d4d",
    width: "100%",
    height: "60px", // Adjust height as needed
  },
};

export default Footer;