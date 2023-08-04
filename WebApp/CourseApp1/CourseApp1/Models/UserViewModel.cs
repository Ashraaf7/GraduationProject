using System;
using System.Collections.Generic;
using System.ComponentModel.DataAnnotations;
using System.Linq;
using System.Web;

namespace CourseApp1.Models
{
    public class RegisterViewModel
    {
        [Required]
        public String Name { get; set; }
        [Required]
        [EmailAddress]
        public String Email { get; set; }
        [Required]
        public String Password { get; set; }
        [Compare("Password", ErrorMessage ="Confirm password doesn't match")]
        public String CofirmPassword { get; set; }
        public String message { get; set; }
    }
    public class LoginViewModel
    {
        [Required]
        [EmailAddress]
        public String Email { get; set; }
        [Required]
        public String Password { get; set; }
        public string ReturnUrl { get; set; }
        public String message { get; set; }
    }
}