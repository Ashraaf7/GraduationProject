using System;
using System.Collections.Generic;
using System.ComponentModel.DataAnnotations;
using System.Linq;
using System.Web;

namespace CourseApp1.Areas.Admin.Data
{
    public class LoginModel
    {
        [EmailAddress]
        [Required]
        [Display(Name ="Email Adress")]
        public string Email { get; set; }

        [DataType(DataType.Password)]
        [Required]
        [Display(Name ="Password")]
        public string Password { get; set; }

        public string message { get; set; }
    }
}